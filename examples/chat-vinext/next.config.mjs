/** @type {import('next').NextConfig} */
const nextConfig = {
  ...(process.env.IGNIS_STATIC_EXPORT === '1'
    ? {
        output: 'export',
        images: { unoptimized: true },
      }
    : {
        async headers() {
          return [
            {
              source: '/(.*)',
              headers: [
                { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
                { key: 'Cross-Origin-Embedder-Policy', value: 'credentialless' },
              ],
            },
          ];
        },
      }),
};

export default nextConfig;
