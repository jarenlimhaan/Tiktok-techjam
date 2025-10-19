import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Location Review Finder",
  description: "Search for locations and find reviews using Mapbox Geocoding",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
