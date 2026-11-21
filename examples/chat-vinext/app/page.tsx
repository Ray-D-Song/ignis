import { getConfiguredModels, isPublicCdnMode } from '@/lib/model-paths';
import { ChatWorkspace } from '@/components/ChatWorkspace';

export default function HomePage() {
  const models = getConfiguredModels();
  const isPublicCdn = isPublicCdnMode();
  return <ChatWorkspace models={models} isPublicCdn={isPublicCdn} />;
}
