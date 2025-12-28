#!/usr/bin/env python3
import http.server
import logging
import socketserver
import sys

PORT = 8080

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range, Access-Control-Allow-Private-Network")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        """Handle Range requests for efficient MCAP streaming."""
        range_header = self.headers.get("Range")
        if not range_header or not range_header.startswith("bytes="):
            return super().do_GET()

        try:
            path = self.translate_path(self.path)
            import os
            file_size = os.path.getsize(path)
            
            # Parse Range: bytes=start-end
            parts = range_header.replace("bytes=", "").split("-")
            start = int(parts[0])
            end = int(parts[1]) if parts[1] else file_size - 1
            
            if start >= file_size:
                self.send_error(416, "Requested Range Not Satisfiable")
                return

            length = end - start + 1
            
            with open(path, "rb") as f:
                f.seek(start)
                content = f.read(length)

            self.send_response(206) # Partial Content
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            self.wfile.write(content)
            
        except Exception as e:
            self.send_error(500, str(e))

def run_server():
    logging.basicConfig(level=logging.INFO)
    handler = CORSRequestHandler
    
    # Allow address reuse to avoid "Address already in use" errors on restart
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"\n================================================================================")
        print(f"🚀 Foxglove MCAP Server started at http://localhost:{PORT}")
        print(f"Serving files from: {sys.path[0]}/..")
        print(f"Press Ctrl+C to stop.")
        print(f"================================================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.server_close()

if __name__ == "__main__":
    run_server()
