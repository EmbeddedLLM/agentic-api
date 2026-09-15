import { refreshCliDocs } from './refresh-cli-docs.mjs';
import { refreshRoadmap } from './refresh-roadmap.mjs';

await refreshRoadmap();
await refreshCliDocs();
console.log(
  'Refreshed the roadmap, Python and Rust CLI references, and Markdown exports.',
);
