import type { Context } from 'hono';
import crypto from 'crypto';
import { createQwenStream, RetryableQwenStreamError, getClientConversation, calculateConversationHash, saveClientConversation, getSessionParent } from '../services/qwen.js';
import type { OpenAIRequest } from '../utils/types.js';
import { getModelContextWindow } from '../core/model-registry.js'
import { truncateMessages, estimateTokenCount } from '../utils/context-truncation.js';
import { getNextAccount, getNextAvailableAccount, markAccountRateLimited, getAccountCooldownInfo, markAccountInUse, releaseAccountInUse, getInUseAccounts, getAccountsWithCooldownSync } from '../core/account-manager.js';
import { loadAccounts } from '../core/accounts.js';
import { registerStream, removeStream, getStream } from '../core/stream-registry.js';
import { metrics } from '../core/metrics.js'
import { addDashboardLog } from '../core/logger.js';
import {
  getForcedToolName,
  getRecentToolNames,
  selectCandidateTools,
  buildCompactToolManifest,
  buildToolCallContract,
  getToolChoiceMode,
} from './tool-handler.js';
import { handleStreamingResponse, handleNonStreamingResponse } from './stream-handler.js';
import { humanDelay } from '../services/human-behavior.js';

export { getIncrementalDelta } from './sse-parser.js';
export type { DeltaResult } from './sse-parser.js';

export interface ActiveChatState {
  chatId: string;
  parentId: string | null;
}

export const activeChatsByAccount = new Map<string, ActiveChatState>();

export async function chatCompletions(c: Context) {
  try {
    const body: OpenAIRequest = await c.req.json();
    const isStream = body.stream ?? false;
    
    let prompt = '';
    const messages = body.messages || [];
    let systemPrompt = '';
    const pendingMultimodal: Array<Array<{ type: string; text?: string; image_url?: { url: string }; video_url?: { url: string }; audio_url?: { url: string }; file_url?: { url: string } }>> = [];

    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      let contentStr = '';
      if (Array.isArray(msg.content)) {
        const textParts: string[] = [];
        const multimodalParts: Array<{ type: string; text?: string; image_url?: { url: string }; video_url?: { url: string }; audio_url?: { url: string }; file_url?: { url: string } }> = [];
        
        for (const p of msg.content as any[]) {
          if (p.type === "text" && p.text) {
            textParts.push(p.text);
          } else if (
            (p.type === "image_url" && p.image_url?.url) ||
            (p.type === "video_url" && p.video_url?.url) ||
            (p.type === "audio_url" && p.audio_url?.url) ||
            (p.type === "file_url" && p.file_url?.url)
          ) {
            multimodalParts.push(p);
          }
        }
        
        contentStr = textParts.join("\n");
        if (multimodalParts.length > 0) {
          pendingMultimodal.push(multimodalParts);
        }
      } else if (typeof msg.content === 'object' && msg.content !== null) {
        contentStr = JSON.stringify(msg.content);
      } else {
        contentStr = msg.content || '';
      }

      if (msg.role === 'system') {
        systemPrompt += (contentStr || '') + '\n\n';
      } else if (msg.role === 'user') {
        prompt += `User: ${contentStr || ''}\n\n`;
      } else if (msg.role === 'assistant') {
        let assistantContent = contentStr || '';
        const reasoning = (msg as any).reasoning_content;
        if (reasoning) {
          assistantContent = `<think>\n${reasoning}\n</think>\n${assistantContent}`;
        }
        if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
           for (const tc of msg.tool_calls) {
             const args = tc.function?.arguments;
             let parsedArgs: any = {};
             if (typeof args === 'string') {
               try { parsedArgs = JSON.parse(args); } catch { parsedArgs = {}; }
             } else if (args && typeof args === 'object') {
               parsedArgs = args;
             }
             const payload = { name: tc.function?.name, arguments: parsedArgs };
             const toolCallStr = `\n<tool_call>\n${JSON.stringify(payload)}\n</tool_call>`;
             assistantContent = assistantContent ? assistantContent + toolCallStr : toolCallStr.trim();
           }
        }
        prompt += `Assistant: ${assistantContent.trim()}\n\n`;
      } else if (msg.role === 'tool' || msg.role === 'function') {
        let toolName = msg.name;
        if (!toolName && msg.tool_call_id) {
          for (let j = i - 1; j >= 0; j--) {
            const prevMsg = messages[j];
            if (prevMsg.role === 'assistant' && prevMsg.tool_calls) {
              const call = prevMsg.tool_calls.find(tc => tc.id === msg.tool_call_id);
              if (call) {
                toolName = call.function?.name;
                break;
              }
            }
          }
        }
        prompt += `Tool Response (${toolName || 'tool'}): ${contentStr || ''}\n\n`;
      }
    }

    const bodyAny = body as any;
    const hasTools = Array.isArray(bodyAny.tools) && bodyAny.tools.length > 0;
    const toolChoiceMode = getToolChoiceMode(bodyAny.tool_choice);
    if (hasTools && toolChoiceMode !== 'none') {
      const formattedTools = bodyAny.tools.map((t: any) => {
        if (t.type === 'function') {
          return {
            name: t.function.name,
            description: t.function.description || '',
            parameters: t.function.parameters
          };
        }
        return t;
      });
      const toolsJson = JSON.stringify(formattedTools, null, 2);
      
      systemPrompt += `\n\n# TOOLS AVAILABLE\nYou have access to the following tools:\n${toolsJson}\n\n# TOOL CALLING FORMAT (MANDATORY)\nTo use a tool, you MUST output a JSON object wrapped EXACTLY in <tool_call> tags:\n\n<tool_call>\n{"name": "tool_name", "arguments": {"param_name": "value"}}\n</tool_call>\n\nEXAMPLE OF MULTIPLE TOOL CALLS:\n<tool_call>\n{"name": "read_file", "arguments": {"path": "file1.txt"}}\n</tool_call>\n<tool_call>\n{"name": "read_file", "arguments": {"path": "file2.txt"}}\n</tool_call>\n\nCRITICAL RULES:\n1. ONLY use the tags above for tool calling. NEVER output raw JSON without tags.\n2. You can call multiple tools by outputting multiple <tool_call> blocks consecutively.\n3. Do NOT output any other text (explanations, chat, etc.) after your <tool_call> blocks. Wait for the user to provide the tool response.\n4. The JSON inside the tags MUST be valid and include ALL required braces and the "arguments" field.\n5. If you need to use a tool, do it IMMEDIATELY without preamble.\n6. NEVER invent, guess, or hallucinate tool names. You MUST ONLY use the exact tool names provided in the 'TOOLS AVAILABLE' list above. Calling an unlisted tool will result in a hard execution error.\n\n`;
      
      if (bodyAny.tool_choice && typeof bodyAny.tool_choice === 'object' && bodyAny.tool_choice.function) {
        const forcedTool = bodyAny.tool_choice.function.name;
        systemPrompt += `CRITICAL: You MUST call the tool "${forcedTool}" in this response.\n\n`;
      }
    }

    const modelId = body.model.replace('-no-thinking', '');
    const modelContextWindow = getModelContextWindow(modelId)
    const estimatedTokens = estimateTokenCount(systemPrompt + prompt, modelId);
    const forcedToolName = getForcedToolName(bodyAny.tool_choice);
    const parallelToolCalls = bodyAny.parallel_tool_calls !== false && toolChoiceMode !== 'forced';
    const toolContextText = `${systemPrompt}\n${prompt}`;
    const recentToolNames = hasTools ? getRecentToolNames(messages) : new Set<string>();
    const candidateTools = hasTools ? selectCandidateTools(bodyAny.tools, toolContextText, forcedToolName, recentToolNames) : [];
    
    let finalPrompt: string;
    if (estimatedTokens > modelContextWindow - 1000) {
      const truncated = truncateMessages(messages, modelContextWindow, systemPrompt, modelId);
      const truncatedBody = truncated.map(m => `${m.role === 'user' ? 'User' : m.role === 'assistant' ? 'Assistant' : m.role}: ${m.content}`).join('\n\n');
      finalPrompt = systemPrompt ? `${systemPrompt}\n\n${truncatedBody}` : truncatedBody;
    } else {
      finalPrompt = systemPrompt ? `${systemPrompt}\n${prompt}` : prompt;
    }

    if (hasTools && toolChoiceMode === 'none') {
      finalPrompt += '\n\n[TOOL USE DISABLED]\nDo not call tools in this response. Answer directly using available context.';
    }

    if (hasTools && toolChoiceMode !== 'none') {
      const compactManifest = buildCompactToolManifest(candidateTools, forcedToolName);
      const toolContract = buildToolCallContract(candidateTools, forcedToolName, parallelToolCalls);
      finalPrompt += `\n\n${toolContract}`;
      if (compactManifest) finalPrompt += `\n\n${compactManifest}`;
    }

    const isThinkingModel = !body.model.includes('no-thinking');

    const isGuestModeOnly = process.env.QWEN_GUEST_MODE_ONLY?.toLowerCase() === 'true';
    let stream: ReadableStream | undefined;
    let uiSessionId = '';
    const completionId = 'chatcmpl-' + crypto.randomUUID();
    let lastError: any = null;

    let isStateful = false;
    let targetAccountId = '';
    let forcedChatId: string | null = null;
    let forcedParentId: string | null = null;

    if (!isGuestModeOnly && messages.length > 1) {
      const previousMessages = messages.slice(0, -1);
      const cachedConv = getClientConversation(previousMessages);

      if (cachedConv) {
        const cooldownInfo = getAccountCooldownInfo(cachedConv.accountId);
        if (!cooldownInfo) {
          const activeAccounts = getAccountsWithCooldownSync();
          
          // Verificar si el usuario cambió la prioridad/cuenta preferencial activa en el dashboard
          const preferredAccount = activeAccounts[0];
          const isPreferredChanged = preferredAccount && cachedConv.accountId !== preferredAccount.id;

          if (!isPreferredChanged) {
            const matchedAccount = activeAccounts.find((a: any) => a.id === cachedConv.accountId);
            if (matchedAccount && !getInUseAccounts().includes(cachedConv.accountId)) {
              isStateful = true;
              targetAccountId = cachedConv.accountId;
              forcedChatId = cachedConv.chatId;
              forcedParentId = cachedConv.parentId;
            }
          } else {
            console.log(`[Chat] Cambio de cuenta activa detectado (${cachedConv.accountId} -> ${preferredAccount.id}). Forzando reinicio de chat en la cuenta preferente.`);
          }
        }
      }
    }

    let statefulPrompt = '';
    if (isStateful) {
      const lastMsg = messages[messages.length - 1];
      if (lastMsg) {
        let contentStr = '';
        if (Array.isArray(lastMsg.content)) {
          const textParts = (lastMsg.content as any[]).filter(p => p.type === 'text').map(p => p.text || '');
          contentStr = textParts.join('\n');
        } else if (typeof lastMsg.content === 'object' && lastMsg.content !== null) {
          contentStr = JSON.stringify(lastMsg.content);
        } else {
          contentStr = lastMsg.content || '';
        }
        
        if (lastMsg.role === 'user') {
          statefulPrompt = `User: ${contentStr}\n\n`;
        } else if (lastMsg.role === 'assistant') {
          statefulPrompt = `Assistant: ${contentStr}\n\n`;
        } else {
          statefulPrompt = `${contentStr}\n\n`;
        }
      }

      if (hasTools && toolChoiceMode === 'none') {
        statefulPrompt += '\n\n[TOOL USE DISABLED]\nDo not call tools in this response. Answer directly using available context.';
      }
      if (hasTools && toolChoiceMode !== 'none') {
        const compactManifest = buildCompactToolManifest(candidateTools, forcedToolName);
        const toolContract = buildToolCallContract(candidateTools, forcedToolName, parallelToolCalls);
        statefulPrompt += `\n\n${toolContract}`;
        if (compactManifest) statefulPrompt += `\n\n${compactManifest}`;
      }
    }

    if (isStateful && targetAccountId && forcedChatId) {
      console.log(`[Chat] Attempting Stateful Chat for session ${forcedChatId} on account ${targetAccountId}`);
      try {
        markAccountInUse(targetAccountId);
        const delayMs = humanDelay(500, 1200);
        await new Promise(resolve => setTimeout(resolve, delayMs));
        const result = await createQwenStream(
          statefulPrompt,
          isThinkingModel,
          body.model,
          forcedParentId,
          targetAccountId === 'global' ? undefined : targetAccountId,
          undefined,
          pendingMultimodal.length > 0 ? pendingMultimodal : undefined,
          forcedChatId
        );
        stream = result.stream;
        uiSessionId = result.uiSessionId;
        registerStream(completionId, {
          abortController: result.controller,
          accountId: result.accountId,
          uiSessionId: result.uiSessionId,
          targetResponseId: '',
          headers: result.headers,
        });
      } catch (err: any) {
        releaseAccountInUse(targetAccountId);
        console.warn(`[Chat] Stateful attempt failed for chat ${forcedChatId}: ${err.message}. Falling back to Stateless...`);
        isStateful = false;
        stream = undefined;
      }
    }

    if (!stream) {
      if (isGuestModeOnly) {
        console.log('[Chat] Modo invitado activo. Omitiendo rotación de cuentas.');
      try {
        const result = await createQwenStream(
          finalPrompt,
          isThinkingModel,
          body.model,
          null,
          'guest',
          undefined,
          pendingMultimodal.length > 0 ? pendingMultimodal : undefined
        );
        stream = result.stream;
        uiSessionId = result.uiSessionId;
        registerStream(completionId, {
          abortController: result.controller,
          accountId: 'guest',
          uiSessionId: result.uiSessionId,
          targetResponseId: '',
          headers: result.headers,
        });
      } catch (err: any) {
        console.error('[Chat] El modo invitado falló:', err.message);
        throw err;
      }
    } else {
      let account = getNextAccount();
      const triedAccountIds = new Set<string>();

      if (!account) {
        let waitTime = 0;
        const maxWaitTime = 20000; // Esperar hasta 20 segundos
        const interval = 1000;     // Comprobar cada 1 segundo
        
        while (!account && waitTime < maxWaitTime) {
          console.log(`[Chat] Todas las cuentas están ocupadas. Esperando ${interval}ms para liberar un carril de cuenta... (${waitTime}/${maxWaitTime}ms)`);
          await new Promise(resolve => setTimeout(resolve, interval));
          waitTime += interval;
          account = getNextAccount();
        }
      }

      if (!account) {
        const inUse = getInUseAccounts();
        const message = inUse.length > 0
          ? `Todos los carriles de cuenta configurados están ocupados: ${inUse.join(', ')}`
          : 'No hay carriles de cuenta disponibles';
        throw new RetryableQwenStreamError(message, 1000);
      }

      while (account) {
        const accountId = account.id;
        const accountEmail = account.email;

        if (triedAccountIds.has(accountId)) {
          account = getNextAvailableAccount(triedAccountIds);
          continue;
        }
        triedAccountIds.add(accountId);

        const cooldownInfo = getAccountCooldownInfo(accountId);
        if (cooldownInfo && accountId !== 'global') {
          console.log(`[Chat] Omitiendo cuenta ${accountEmail} (${accountId}) — en cooldown por ${Math.round(cooldownInfo.remainingMs / 1000)}s (${cooldownInfo.reason})`);
          account = getNextAvailableAccount(triedAccountIds);
          continue;
        }

        addDashboardLog(`Enrutando petición a la cuenta: ${accountEmail} (${accountId})`, 'info', 'Chat');
        markAccountInUse(accountId);

        // --- REUTILIZACIÓN DE CONVERSACIÓN ACTIVA (STATEFUL) ---
        let isStatefulForAccount = false;
        let forcedChatIdForAccount: string | null = null;
        let forcedParentIdForAccount: string | null = null;

        const activeChat = activeChatsByAccount.get(accountId);
        if (activeChat) {
          isStatefulForAccount = true;
          forcedChatIdForAccount = activeChat.chatId;
          forcedParentIdForAccount = activeChat.parentId;
          addDashboardLog(`Reutilizando chat activo ${forcedChatIdForAccount} con padre ${forcedParentIdForAccount} para la cuenta ${accountEmail}`, 'info', 'Chat');
        }

        // Construir el prompt específico (stateful o completo)
        let promptForAccount = finalPrompt;
        if (isStatefulForAccount) {
          const lastMsg = messages[messages.length - 1];
          if (lastMsg) {
            let contentStr = '';
            if (Array.isArray(lastMsg.content)) {
              const textParts = (lastMsg.content as any[]).filter(p => p.type === 'text').map(p => p.text || '');
              contentStr = textParts.join('\n');
            } else if (typeof lastMsg.content === 'object' && lastMsg.content !== null) {
              contentStr = JSON.stringify(lastMsg.content);
            } else {
              contentStr = lastMsg.content || '';
            }
            
            if (lastMsg.role === 'user') {
              promptForAccount = `User: ${contentStr}\n\n`;
            } else if (lastMsg.role === 'assistant') {
              promptForAccount = `Assistant: ${contentStr}\n\n`;
            } else {
              promptForAccount = `${contentStr}\n\n`;
            }
          }

          if (hasTools && toolChoiceMode === 'none') {
            promptForAccount += '\n\n[TOOL USE DISABLED]\nDo not call tools in this response. Answer directly using available context.';
          }
          if (hasTools && toolChoiceMode !== 'none') {
            const compactManifest = buildCompactToolManifest(candidateTools, forcedToolName);
            const toolContract = buildToolCallContract(candidateTools, forcedToolName, parallelToolCalls);
            promptForAccount += `\n\n${toolContract}`;
            if (compactManifest) promptForAccount += `\n\n${compactManifest}`;
          }
        }

        let retries = 3;
        let retryDelay = 500;
        let success = false;

        try {
          while (retries > 0) {
            try {
              const result = await createQwenStream(
                promptForAccount,
                isThinkingModel,
                body.model,
                isStatefulForAccount ? forcedParentIdForAccount : null,
                accountId === 'global' ? undefined : accountId,
                undefined,
                pendingMultimodal.length > 0 ? pendingMultimodal : undefined,
                isStatefulForAccount ? forcedChatIdForAccount : null
              );
              stream = result.stream;
              uiSessionId = result.uiSessionId;
              registerStream(completionId, {
                abortController: result.controller,
                accountId: result.accountId,
                uiSessionId: result.uiSessionId,
                targetResponseId: '',
                headers: result.headers,
              });
              success = true;
              break;
            } catch (err: any) {
              retries--;

              // Si ocurrió un error usando el chat activo de la cuenta, lo borramos y reintentamos en modo stateless
              if (isStatefulForAccount) {
                console.warn(`[Chat] La generación con estado falló en el chat activo ${forcedChatIdForAccount} para la cuenta ${accountEmail}: ${err.message}. Limpiando el chat activo y reintentando sin estado...`);
                activeChatsByAccount.delete(accountId);
                isStatefulForAccount = false;
                forcedChatIdForAccount = null;
                forcedParentIdForAccount = null;
                promptForAccount = finalPrompt; // Restaurar el prompt completo original
              }

              if (err.upstreamCode === 'RateLimited' || err.upstreamStatus === 429) {
                const hourHint = err.message?.match(/Wait about (\d+) hour/);
                const hours = hourHint ? parseInt(hourHint[1]) : 24;
                const cooldownMs = hours * 60 * 60 * 1000;
                markAccountRateLimited(accountId, cooldownMs, 'RateLimited');
                console.warn(`[Chat] Cuenta ${accountEmail} (${accountId}) con límite de tasa activa. Entrando en cooldown por ${hours} horas.`);
                lastError = err;
                break;
              }

              if (retries === 0) {
                if (err.upstreamStatus && err.upstreamStatus >= 500) {
                  markAccountRateLimited(accountId, undefined, 'ServerError');
                  console.warn(`[Chat] Cuenta ${accountEmail} (${accountId}) devolvió un error de servidor. Marcada para entrar en cooldown.`);
                }
                lastError = err;
                break;
              }

              let useDelay = retryDelay;
              if (err instanceof RetryableQwenStreamError && err.retryAfterMs !== undefined) {
                useDelay = err.retryAfterMs;
              }
              const isRetryable = err instanceof RetryableQwenStreamError || err.message?.includes('in progress') || err.message?.includes('Bad_Request');
              if (!isRetryable) {
                lastError = err;
                break;
              }
              console.warn(`[Chat] Petición a Qwen falló para ${accountEmail}, reintentando en ${useDelay}ms... (quedan ${retries} reintentos)`);
              await new Promise(r => setTimeout(r, useDelay));
              retryDelay = Math.min(retryDelay * 2, 5000);
            }
          }
        } finally {
          if (!success) {
            releaseAccountInUse(accountId);
          }
        }

        if (success) {
          break;
        }

        account = getNextAvailableAccount(triedAccountIds);
      }
    }
    }

    if (!stream) {
      removeStream(completionId);
      const accounts = loadAccounts();
      const allOnCooldown = accounts.length === 0 || accounts.every(a => getAccountCooldownInfo(a.id) !== null);
      
      if (allOnCooldown) {
        console.warn(`[Chat] CRITICAL: All accounts are rate-limited, on cooldown, or none configured! Falling back to GUEST mode.`);
        try {
          const result = await createQwenStream(
            finalPrompt,
            isThinkingModel,
            body.model,
            null,
            'guest',
            undefined,
            pendingMultimodal.length > 0 ? pendingMultimodal : undefined
          );
          stream = result.stream;
          uiSessionId = result.uiSessionId;
          registerStream(completionId, {
            abortController: result.controller,
            accountId: 'guest',
            uiSessionId: result.uiSessionId,
            targetResponseId: '',
            headers: result.headers,
          });
        } catch (guestErr: any) {
          console.error('[Chat] Guest mode also failed:', guestErr.message);
          throw lastError || new Error('All accounts and guest mode failed');
        }
      } else {
        throw lastError || new Error('All accounts failed');
      }
    }

    if (!isStream) {
      return handleNonStreamingResponse(
        c,
        stream!,
        completionId,
        body.model,
        uiSessionId,
        hasTools && toolChoiceMode !== 'none',
        bodyAny.tools || [],
        getStream(completionId)?.accountId || targetAccountId || 'guest',
        messages
      );
    }

    return handleStreamingResponse(c, {
      stream: stream!,
      completionId,
      model: body.model,
      uiSessionId,
      hasTools: hasTools && toolChoiceMode !== 'none',
      tools: bodyAny.tools || [],
      finalPrompt,
      streamOptions: body.stream_options,
      accountId: getStream(completionId)?.accountId || targetAccountId || 'guest',
      messages
    });
  } catch (err: any) {
    console.error('Error in chatCompletions:', err)
    const status = err.upstreamStatus || 500
    if (status >= 500) {
      metrics.increment('requests.errors')
    }
    return c.json({ error: { message: err.message } }, status)
  }
}

export async function chatCompletionsStop(c: Context) {
  try {
    const body = await c.req.json();
    const { chat_id, response_id } = body;

    if (!chat_id || !response_id) {
      return c.json({ error: 'chat_id and response_id are required' }, 400);
    }

    const stream = getStream(chat_id);
    if (!stream) {
      return c.json({ error: 'Stream not found' }, 404);
    }

    if (stream.targetResponseId && stream.targetResponseId !== response_id) {
      return c.json({ error: 'response_id mismatch' }, 400);
    }

    const stopResponse = await fetch(`https://chat.qwen.ai/api/v2/chat/completions/stop?chat_id=${chat_id}`, {
      method: 'POST',
      headers: {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'pt-BR,pt;q=0.9',
        'Content-Type': 'application/json',
        'Cookie': stream.headers.cookie,
        'Origin': 'https://chat.qwen.ai',
        'Referer': `https://chat.qwen.ai/c/${chat_id}`,
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': stream.headers['user-agent'],
        'X-Request-Id': crypto.randomUUID(),
        'bx-ua': stream.headers['bx-ua'],
        'bx-umidtoken': stream.headers['bx-umidtoken'],
        'bx-v': stream.headers['bx-v'],
      },
      body: JSON.stringify({ chat_id, response_id }),
    });

    if (!stopResponse.ok) {
      const errorText = await stopResponse.text();
      console.error(`[Stop] Failed to stop generation for chat_id=${chat_id}: ${stopResponse.status} ${errorText}`);
      return c.json({ error: 'Failed to stop generation' }, stopResponse.status as any);
    }

    stream.abortController.abort();
    removeStream(chat_id);

    console.log(`[Stop] Generation stopped for chat_id=${chat_id}`);
    return c.json({ success: true });
  } catch (err: any) {
    console.error('Error in chatCompletionsStop:', err);
    return c.json({ error: err.message }, 500);
  }
}
