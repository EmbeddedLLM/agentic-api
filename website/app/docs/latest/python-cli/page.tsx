import type { Metadata } from 'next';
import { CliReferencePage } from '@/components/site/cli-reference-page';
import reference from '@/content/python-cli.md?raw';

export const metadata: Metadata = {
  title: 'Python command-line reference',
  description:
    'Commands, options, and defaults for the vLLM Agentic API Python command-line interface (agentic-api): serve, doctor, and version. Generated from the CLI source.',
  alternates: { canonical: '/docs/latest/python-cli' },
};

export default function PythonCliReference() {
  return <CliReferencePage kind="python" reference={reference} />;
}
