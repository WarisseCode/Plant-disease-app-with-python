from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

# Recherches multiples
arguments = {
    "keywords": "cars, buildings, animals, tools",
    "limit": 1000,
    "print_urls": True,
    "output_directory": "dataset/other"
}
response.download(arguments)
