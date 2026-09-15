import { CalendarDays, FileText } from 'lucide-react';
import { COMMUNITY_MEETING, COMMUNITY_NOTES } from '@/lib/site';

export function CommunityMeetingLinks() {
  return (
    <div className="community-meeting-links">
      <a className="text-link" href={COMMUNITY_MEETING}>
        <CalendarDays size={16} aria-hidden="true" /> Community meeting invite
      </a>
      <a className="text-link" href={COMMUNITY_NOTES}>
        <FileText size={16} aria-hidden="true" /> Meeting notes
      </a>
    </div>
  );
}
