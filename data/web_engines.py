from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from func_timeout import FunctionTimedOut, func_set_timeout
import requests


class WebEngine():
    def get_page_source(self, url: str) -> str:
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass


MAC_HEADERS = {
    'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.67',
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
}

class RequestsWebEngine(WebEngine):
    def __init__(self, headers: str = MAC_HEADERS):
        self.headers = headers

    @func_set_timeout(10)
    def _request_get_run(self, url: str) -> str:
        resp = requests.get(url, headers=self.headers)

        status_code = resp.status_code
        if status_code >= 300:
            print('status code:', status_code)
            return None
        
        return resp.text

    def request_get(self, url: str) -> str:
        try:
            return self._request_get_run(url)
        except FunctionTimedOut as e:
            print(e)
            return None


class SeleniumWebEngine(WebEngine):
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run Chrome in headless mode (no GUI)
        chrome_options.add_argument('--no-sandbox')  # Bypass OS security model
        chrome_options.add_argument('--disable-dev-shm-usage')  # Avoids issues with Docker
        chrome_options.add_argument('--window-size=1920x1080')
        chrome_options. add_argument('--blink-settings=imagesEnabled=false')

        self.driver = webdriver.Chrome(options=chrome_options)

    @func_set_timeout(20)
    def _request_get_run(self, url: str) -> str:
        self.driver.get(url)

        return self.driver.page_source

    def request_get(self, url: str) -> str:
        try:
            return self._request_get_run(url)
        except FunctionTimedOut as e:
            print(e)
            return None

    def __exit__(self, type, value, traceback):
        self.driver.close()
