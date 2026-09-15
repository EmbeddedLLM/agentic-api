import type { Metadata } from 'next';
import { CliReferencePage } from '@/components/site/cli-reference-page';
import reference from '@/content/rust-cli.md?raw';

export const metadata: Metadata = {
  title: 'Rust command-line reference',
  description:
    'Commands, options, and defaults for the agentic Rust CLI: run Codex or Claude Code, attach a harness, serve, and validate. Generated from the CLI source.',
  alternates: { canonical: '/docs/latest/rust-cli' },
};

export default function RustCliReference() {
  return <CliReferencePage kind="rust" reference={reference} />;
}
