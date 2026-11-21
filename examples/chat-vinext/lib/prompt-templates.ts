import type { ModelPreset } from './model-catalog';

export type ArchKind = 'spark' | 'lfm3' | 'gemma4';
export type OmniGenMode = 'text' | 'interleaved' | 'audio';

export interface SimpleChatTurn {
  role: 'user' | 'assistant';
  content: string;
  hasImage?: boolean;
  hasAudio?: boolean;
}

export function getArchKind(model?: ModelPreset): ArchKind {
  if (!model) return 'spark';
  if (model.topModelType === 'lfm3_omni' || model.architecture === 'lfm3') return 'lfm3';
  if (model.topModelType === 'gemma4' || model.architecture === 'gemma4_text') return 'gemma4';
  return 'spark';
}

export function splitThinking(raw: string): { think: string; answer: string; isThinking: boolean } {
  const pairs: Array<[string, string]> = [
    ['<think>', '</think>'],
    ['<|thought|>', '<thought|>'],
  ];
  for (const [startTag, endTag] of pairs) {
    const sIdx = raw.indexOf(startTag);
    const eIdx = raw.indexOf(endTag);
    if (sIdx !== -1 && eIdx !== -1 && eIdx >= sIdx) {
      return {
        think: raw.slice(sIdx + startTag.length, eIdx).trim(),
        answer: raw.slice(eIdx + endTag.length).trimStart(),
        isThinking: false,
      };
    }
    if (sIdx !== -1 && eIdx === -1) {
      return {
        think: raw.slice(sIdx + startTag.length).trimStart(),
        answer: '',
        isThinking: true,
      };
    }
    if (eIdx !== -1) {
      return {
        think: raw.slice(0, eIdx).trim(),
        answer: raw.slice(eIdx + endTag.length).trimStart(),
        isThinking: false,
      };
    }
  }
  return { think: '', answer: raw, isThinking: false };
}

export function buildSparkSystemPrefix(
  enableThinking: boolean,
  effort: 'low' | 'high',
  customSystemPrompt?: string,
): string {
  const baseSys = customSystemPrompt || 'you are a helpful assistant.';
  const reasoning = enableThinking
    ? `\n\`thinking_effort\` guides on how much to think in your thinking channel (not including the response channel), supported values include \`low\` and \`high\`.\nNow the system is invoked with \`thinking_effort=${effort}\`.`
    : '';
  return `<｜start▁of▁sentence｜><|System|>\n${baseSys}${reasoning}<｜end▁of▁sentence｜>`;
}

export function formatSingleTurnPrompt(
  archKind: ArchKind,
  opts: {
    systemPrompt?: string;
    userPrompt: string;
    enableThinking?: boolean;
    effort?: 'low' | 'high';
    includeImageTag?: boolean;
    includeAudioTag?: boolean;
  },
): string {
  const {
    systemPrompt,
    userPrompt,
    enableThinking = false,
    effort = 'low',
    includeImageTag = false,
    includeAudioTag = false,
  } = opts;

  if (archKind === 'spark') {
    const sys = buildSparkSystemPrefix(enableThinking, effort, systemPrompt);
    return `${sys}<｜start▁of▁sentence｜><|User|>${userPrompt}<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>${
      enableThinking ? '<think>' : '</think>'
    }`;
  }

  if (archKind === 'lfm3') {
    let mm = '';
    if (includeImageTag) mm += '<|vision_start|><|image|><|vision_end|>';
    if (includeAudioTag) mm += '<|vision_start|><|audio|><|vision_end|>';
    const sysBlock = systemPrompt ? `<|im_start|>system\n${systemPrompt}<|im_end|>\n` : '';
    return `<|startoftext|>${sysBlock}<|im_start|>user\n${mm}${userPrompt}<|im_end|>\n<|im_start|>assistant\n<think></think>`;
  }

  let mm = '';
  if (includeImageTag) mm += '<|image|>';
  if (includeAudioTag) mm += '<|audio|>';
  const sysPrefix = systemPrompt ? `${systemPrompt}\n\n` : '';
  return `<bos><|turn>user\n${mm}${sysPrefix}${userPrompt}<turn|>\n${
    enableThinking ? '<|turn>model\n<|thought|>\n' : '<|turn>model\n'
  }`;
}
