import requests
import time


class Website:
    def __init__(self, url):
        self.status = None
        self.url = url
        self.z = 20

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

    def __repr__(self):
        return f"{self.__class__.__name__}({self.url}) ,{self.__dict__} "


youtube = Website("https://google.com")

# for _ in range(5):
#     print(youtube.get_status())
#     print(youtube.is_changed())


# This is same as  youtube.z
print(getattr(youtube, "z"))

# if the attribute isn't exist you can get default value
print(getattr(youtube, "z", 10))


# set attribute or delattribute is to create or delete attributes of objects in runtime
setattr(youtube, "z", 30)
print(youtube.z)

delattr(youtube, "z")  # delete attributes of objects
# print(youtube.z)  # AttributeError: 'Website' object has no attribute 'z'


# we can use dictionary of attributes

website_dict = {
    "users": 20,
    "videos": 100
}

for k, v in website_dict.items():
    setattr(youtube, k, v)
    print(f"{k}: {getattr(youtube, k)}")


print(youtube)
