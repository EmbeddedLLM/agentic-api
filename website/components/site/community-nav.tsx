'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  ArrowUpRight,
  Users,
  HeartHandshake,
  CalendarDays,
  FileText,
} from 'lucide-react';
import { REPO, COMMUNITY_MEETING, COMMUNITY_NOTES } from '@/lib/site';
export function CommunityNav() {
  const path = usePathname();
  return (
    <nav className="community-nav" aria-label="Community navigation">
      <div>
        <Link
          href="/community/team"
          aria-current={
            path.replace(/\/$/, '') === '/community/team' ? 'page' : undefined
          }
        >
          <Users size={16} /> Project team
        </Link>
        <Link
          href="/community/contributors"
          aria-current={
            path.replace(/\/$/, '') === '/community/contributors'
              ? 'page'
              : undefined
          }
        >
          <HeartHandshake size={17} /> Contributors
        </Link>
        <a href={COMMUNITY_MEETING}>
          <CalendarDays size={16} aria-hidden="true" /> Meeting invite
        </a>
        <a href={COMMUNITY_NOTES}>
          <FileText size={16} aria-hidden="true" /> Meeting notes
        </a>
      </div>
      <a href={`${REPO}/blob/main/CONTRIBUTING.md`}>
        Contributing guide <ArrowUpRight size={14} />
      </a>
    </nav>
  );
}
