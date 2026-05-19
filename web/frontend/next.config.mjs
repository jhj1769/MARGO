/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "images.unsplash.com" },
      { protocol: "https", hostname: "m.media-amazon.com" }
    ]
  },
  async rewrites() {
    const api = process.env.MARGO_API_URL || "http://localhost:8001";
    return [
      { source: "/api/margo/:path*", destination: `${api}/api/:path*` },
      { source: "/ws/margo/:path*",  destination: `${api}/ws/:path*` }
    ];
  }
};

export default nextConfig;
