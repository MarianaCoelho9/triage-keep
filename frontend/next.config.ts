import type { NextConfig } from "next";

const backendOrigin = process.env.BACKEND_ORIGIN ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  reactCompiler: true,
  async rewrites() {
    return [
      {
        source: "/report",
        destination: `${backendOrigin}/report`,
      },
      {
        source: "/report/fhir",
        destination: `${backendOrigin}/report/fhir`,
      },
      {
        source: "/static/:path*",
        destination: `${backendOrigin}/static/:path*`,
      },
    ];
  },
};

export default nextConfig;
