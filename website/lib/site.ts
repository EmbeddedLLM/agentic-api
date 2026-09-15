export const REPO = 'https://github.com/vllm-project/agentic-api';
export const DOCS = '/docs';
export const SLACK = 'https://slack.vllm.ai/';
export const SLACK_CHANNEL = '#sig-agentic-api';
export const COMMUNITY_MEETING =
  'https://zoom-lfx.platform.linuxfoundation.org/meeting/99957052764?password=caffe82c-cdfe-478e-964c-5e1d95c0c72e&invite=true';
export const COMMUNITY_NOTES =
  'https://docs.google.com/document/d/1zJUfxdloxu9_oBYfbfTtgbz5sSZxwAnqvlnt72T67oU/edit?tab=t.0#heading=h.d5de0vs6frfk';
export const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ||
  'https://vllm-agentic-api.franciscojavierarceo.chatgpt.site';

export function assetPath(path: string) {
  return `${process.env.NEXT_PUBLIC_BASE_PATH || ''}${path}`;
}
