import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

@pytest.fixture(scope="module")
def browser():
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service
    
    options = webdriver.ChromeOptions()
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    driver.maximize_window()
    yield driver
    driver.quit() 

def test_streamlit_dashboard_loads(browser):
    print("\n[INFO] Opening Streamlit Dashboard...")
    browser.get("http://localhost:8501")
    
    WebDriverWait(browser, 15).until(
        EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'IntelliPark')]"))
    )
    assert "IntelliPark" in browser.page_source
    time.sleep(2) 

def test_navigation_to_anpr(browser):
    print("[INFO] Clicking the ANPR Alert Button...")
    anpr_button = WebDriverWait(browser, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[descendant::p[contains(text(), 'VIEW LIVE ANPR ALERTS')]] | //button[contains(., 'VIEW LIVE ANPR ALERTS')]"))
    )
    anpr_button.click()
    
    WebDriverWait(browser, 15).until(
        EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'ANPR Gateway Sentinel')]"))
    )
    assert "ANPR Gateway Sentinel" in browser.page_source
    time.sleep(4) 

def test_navigation_back_to_dashboard(browser):
    print("[INFO] Returning to Dashboard...")
    dash_button = WebDriverWait(browser, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[descendant::p[contains(text(), 'BACK TO DASHBOARD')]] | //button[contains(., 'BACK TO DASHBOARD')]"))
    )
    dash_button.click()
    
    WebDriverWait(browser, 15).until(
        EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Multi-Modal Context-Aware System')]"))
    )
    assert "Multi-Modal Context-Aware System" in browser.page_source
    time.sleep(2)