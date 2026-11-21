import { discoverIgnisModels } from '@/lib/model-paths';
import { ChatWorkspace } from '@/components/ChatWorkspace';

export const dynamic = 'force-dynamic';

export default function HomePage() {
  const models = discoverIgnisModels();
  return <ChatWorkspace models={models} />;
}
