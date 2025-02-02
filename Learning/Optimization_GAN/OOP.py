import requests
import time


class Website:
    def __init__(self, url):
        self.status = None
        self.url = url

    def get_status(self):
        if self.url:  # Ensure URL is not None
            self.status = requests.get(self.url).status_code
        return self.status

    def is_changed(self):
        status = requests.get(self.url).status_code
        if self.status == None:
            self.status = status
            return False
        elif self.status != status:
            self.status = status
            return True
        else:
            # This means the status didn't change
            return False


youtube = Website("https://google.com")

for _ in range(5):
    time.sleep(1)
    print(youtube.get_status())
    print(youtube.is_changed())
