/** @type {import('next').NextConfig} */
const nextConfig = {
  ...(process.env.IGNIS_STATIC_EXPORT === '1'
    ? {
        output: 'export',
        images: { unoptimized: true },
      }
    : {}),
};

export default nextConfig;
