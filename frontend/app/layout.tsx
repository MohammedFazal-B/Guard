import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LLM Guard",
  description: "Protected local chat with Kev and Gemma",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className="dark h-full antialiased"
      style={{ fontFamily: "Arial, Helvetica, sans-serif" }}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
