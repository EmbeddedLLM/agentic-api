import type { Metadata } from 'next';
import { defaultUrlTransform } from 'react-markdown';
import { LinkedMarkdown } from '@/components/site/linked-markdown';
import { ArrowUpRight, FileText, GitBranch } from 'lucide-react';
import { CommunityCTA } from '@/components/site/community-cta';
import { REPO, assetPath } from '@/lib/site';
import roadmap from '@/content/roadmap.md?raw';

export const metadata: Metadata = {
  title: 'Roadmap',
  description:
    'Explore the vLLM Agentic API roadmap: stateful APIs, Codex support, built-in tool execution, and production readiness.',
  alternates: { canonical: '/roadmap' },
};

function roadmapUrl(url: string) {
  const safe = defaultUrlTransform(url);
  if (!safe || safe.startsWith('#') || /^(?:[a-z]+:)?\/\//i.test(safe))
    return safe;
  if (/^[a-z]+:/i.test(safe)) return safe;
  return new URL(safe, REPO + '/blob/main/').href;
}

export default function Roadmap() {
  // The page supplies its own h1; retain the rest of the repository document.
  const body = roadmap.replace(/^# [^\r\n]+\r?\n\s*/, '');
  return (
    <main id="main" className="container roadmap-page">
      <link
        rel="alternate"
        type="text/markdown"
        href={assetPath('/roadmap.md')}
      />
      <header className="community-hero">
        <span className="eyebrow">BUILD / ROADMAP</span>
        <h1>
          The road
          <br />
          <span>ahead.</span>
        </h1>
        <p>
          The priorities shaping vLLM Agentic API, from stateful conversations
          and built-in tools to production infrastructure.
        </p>
      </header>
      <div className="roadmap-source">
        <span>
          <GitBranch size={16} aria-hidden="true" /> Project direction
        </span>
        <div>
          <a href={assetPath('/roadmap.md')}>
            <FileText size={16} aria-hidden="true" /> Read as Markdown
          </a>
          <a href={REPO + '/blob/main/ROADMAP.md'}>
            View on GitHub <ArrowUpRight size={16} aria-hidden="true" />
          </a>
        </div>
      </div>
      <article className="roadmap-body" aria-label="Project roadmap">
        <LinkedMarkdown urlTransform={roadmapUrl}>{body}</LinkedMarkdown>
      </article>
      <CommunityCTA />
    </main>
  );
}
