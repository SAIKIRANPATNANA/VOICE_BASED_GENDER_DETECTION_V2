import time
from playwright.sync_api import sync_playwright

def capture():
    print("Launching Chromium via /usr/bin/google-chrome...")
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 2600})
        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        print("Waiting for Streamlit charts and arena cards to render...")
        page.wait_for_selector(".arena-card", timeout=30000)
        time.sleep(3)
        out_path = "/home/user/.gemini/antigravity-ide/brain/e1919ac1-8b33-45d0-bcc3-63ebe13850a1/streamlit_dashboard_rendered.png"
        page.screenshot(path=out_path, full_page=True)
        print(f"✓ Screenshot saved to {out_path}")
        browser.close()

if __name__ == "__main__":
    capture()
