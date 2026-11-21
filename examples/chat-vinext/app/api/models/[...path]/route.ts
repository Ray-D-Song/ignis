import fs from 'node:fs';
import path from 'node:path';
import { discoverIgnisModels } from '@/lib/model-paths';

export async function GET(
  _req: Request,
  context: { params: Promise<{ path: string[] }> }
) {
  const { path: segments } = await context.params;
  if (!segments || segments.length < 2) {
    return new Response('Invalid model path', { status: 400 });
  }

  const [modelId, ...rest] = segments;
  const models = discoverIgnisModels();
  const model = models.find((m) => m.id === modelId);
  if (!model) {
    return new Response(`Model not found: ${modelId}`, { status: 404 });
  }

  const relPath = rest.join('/');
  const resolved = path.resolve(model.dirPath, relPath);
  if (!resolved.startsWith(path.resolve(model.dirPath) + path.sep) || !fs.existsSync(resolved)) {
    return new Response('File not found', { status: 404 });
  }

  const stat = fs.statSync(resolved);
  if (!stat.isFile()) {
    return new Response('Not a file', { status: 404 });
  }

  const ext = path.extname(resolved);
  const contentType = ext === '.json'
    ? 'application/json; charset=utf-8'
    : 'application/octet-stream';

  const nodeStream = fs.createReadStream(resolved, { highWaterMark: 1024 * 1024 });
  const webStream = new ReadableStream({
    start(controller) {
      nodeStream.on('data', (chunk) => controller.enqueue(new Uint8Array(chunk as Buffer)));
      nodeStream.on('end', () => controller.close());
      nodeStream.on('error', (err) => controller.error(err));
    },
    cancel() {
      nodeStream.destroy();
    },
  });

  return new Response(webStream, {
    status: 200,
    headers: {
      'Content-Type': contentType,
      'Content-Length': String(stat.size),
      'Cache-Control': 'public, max-age=31536000, immutable',
      'Cross-Origin-Resource-Policy': 'same-origin',
    },
  });
}
