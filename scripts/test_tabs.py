import time
from playwright.sync_api import sync_playwright

def test_tabs():
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 2600})
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        page.wait_for_selector(".arena-card", timeout=30000)
        time.sleep(2)
        
        # Click Tab 2: 1D FFT Magnitude Spectrum
        print("Clicking Tab 2 (FFT)...")
        tabs = page.query_selector_all('[data-baseweb="tab"]')
        if len(tabs) >= 2:
            tabs[1].click()
            time.sleep(2)
            page.screenshot(path="/home/user/.gemini/antigravity-ide/brain/e1919ac1-8b33-45d0-bcc3-63ebe13850a1/streamlit_tab2_fft.png")
            
        # Click Tab 3: 2D Log-Mel Spectrogram
        print("Clicking Tab 3 (Mel Spectrogram)...")
        if len(tabs) >= 3:
            tabs[2].click()
            time.sleep(2)
            page.screenshot(path="/home/user/.gemini/antigravity-ide/brain/e1919ac1-8b33-45d0-bcc3-63ebe13850a1/streamlit_tab3_mel.png")
            
        browser.close()
        print("Tabs verified and captured successfully!")

if __name__ == "__main__":
    test_tabs()
