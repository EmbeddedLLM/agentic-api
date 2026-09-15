import Link from 'next/link';
import { defaultUrlTransform } from 'react-markdown';
import { ArrowLeft, ArrowUpRight, FileText, GitBranch } from 'lucide-react';
import { LinkedMarkdown } from './linked-markdown';
import { REPO, assetPath } from '@/lib/site';

const interfaces = {
  rust: {
    label: 'Rust CLI',
    command: 'agentic',
    source: 'crates/agentic-server/src/agentic_cli.rs',
  },
  python: {
    label: 'Python launcher',
    command: 'agentic-api',
    source: 'python/agentic_api/cli.py',
  },
};

function referenceUrl(url: string) {
  const safe = defaultUrlTransform(url);
  if (safe === 'rust-cli.md' || safe === 'python-cli.md')
    return assetPath('/docs/latest/' + safe.slice(0, -3));
  if (!safe || safe.startsWith('#') || /^(?:[a-z]+:)?\/\//i.test(safe))
    return safe;
  if (/^[a-z]+:/i.test(safe)) return safe;
  return new URL(safe, REPO + '/blob/main/docs/reference/').href;
}

export function CliReferencePage({
  kind,
  reference,
}: {
  kind: keyof typeof interfaces;
  reference: string;
}) {
  const cli = interfaces[kind];
  const body = reference.replace(/^# [^\r\n]+\r?\n\s*/, '');
  const markdownPath = assetPath(`/docs/latest/${kind}-cli.md`);
  return (
    <main id="main" className="container roadmap-page">
      <link rel="alternate" type="text/markdown" href={markdownPath} />
      <header className="community-hero">
        <span className="eyebrow">BUILD / COMMAND-LINE INTERFACES</span>
        <h1>
          {kind === 'rust' ? 'Rust CLI.' : 'Python CLI.'}
          <br />
          <span>Command-line reference.</span>
        </h1>
        <p>
          {kind === 'rust' ? (
            <>
              Run <code>agentic</code> in your terminal to start the gateway and
              launch Codex or Claude Code. Available through Cargo and the PyPI
              package.
            </>
          ) : (
            <>
              Run the Python launcher from your terminal with{' '}
              <code>agentic-api</code> or <code>python -m agentic_api</code>.
              Start the gateway, manage local inference, and check your setup.
            </>
          )}
        </p>
      </header>
      <nav className="cli-interface-nav" aria-label="Command-line interface">
        {Object.entries(interfaces).map(([id, item]) => (
          <Link
            key={id}
            href={`/docs/latest/${id}-cli`}
            aria-current={kind === id ? 'page' : undefined}
          >
            <span>{item.label}</span>
            <code>
              <span className="command-prompt" aria-hidden="true">
                $
              </span>
              {item.command}
            </code>
          </Link>
        ))}
      </nav>
      <aside className="docs-version-notice">
        <strong>Latest / development</strong>
        <p>
          Generated from the source at each website build. This reference may
          include changes beyond your installed release. Run{' '}
          <code>{cli.command} --help</code> for the installed CLI, or{' '}
          <Link href="/docs">choose a tagged documentation snapshot</Link>.
        </p>
      </aside>
      <div className="roadmap-source">
        <span>
          <GitBranch size={16} aria-hidden="true" /> Source-generated reference
        </span>
        <div>
          <a href={markdownPath}>
            <FileText size={16} aria-hidden="true" /> Read as Markdown
          </a>
          <a href={REPO + '/blob/main/' + cli.source}>
            View CLI source <ArrowUpRight size={16} aria-hidden="true" />
          </a>
        </div>
      </div>
      <article className="roadmap-body" aria-label={`${cli.label} reference`}>
        <p className="command-prompt-note">
          The <code>$</code> marks a terminal prompt; leave it out when typing
          commands.
        </p>
        <LinkedMarkdown commandHeadings urlTransform={referenceUrl}>
          {body}
        </LinkedMarkdown>
      </article>
      <div className="docs-all-link">
        <Link href="/docs/latest">
          <ArrowLeft size={16} aria-hidden="true" /> All development
          documentation
        </Link>
      </div>
    </main>
  );
}
