import type { NextConfig } from "next";

const apiHost = process.env.NEXT_PUBLIC_API_HOST || "localhost:8080";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // API proxy configuration
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `http://${apiHost}/api/:path*`,
      },
      {
        source: "/ws/:path*",
        destination: `http://${apiHost}/ws/:path*`,
      },
    ];
  },
};

export default nextConfig;
