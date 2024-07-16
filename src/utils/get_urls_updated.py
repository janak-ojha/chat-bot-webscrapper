from urllib.parse import urlparse,urljoin
import requests
from bs4 import BeautifulSoup

def get_links(url):
    try:
        headers ={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response=requests.get(url,headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text,'html.parser')
            links = [link.get('href') for link in soup.find_all('a',href=True)]
            return links
        else:
            print(f"failed to retrive page {url}: {response.status_code}")
            return []
    except Exception as e:
        print(f"An error occured while retrieving the page(url): (str(e))") 
        return []   
    
def normalize_url(url):
    parsed_url = urlparse(url)
    scheme = parsed_url.schema.lower()
    netloc = parsed_url.netloc.lower().replace('www','')
    path = parsed_url.path.rstrip("/")
    normarlize_url = f"{scheme}://{netloc}{path}"
    return normarlize_url

def filter_links(links,base_domain):
    valid_links = []
    for link in links:
        if link is None:
            continue
        parsed_url = urlparse(link)
        noramlized_url = normalize_url(link)
        if base_domain == parsed_url.netloc.lower() or base_domain in parsed_url.path.lower():
            if not any (noramlized_url.lower().endswith(ext) for ext in ('.jpg','.png','.gif','.avi','.mp4','.mp3')):
                valid_links.append(noramlized_url)
    return valid_links

def scrape_website(url,base_domain,depth,visited = None):
    if visited is None:
        visited = set()
    if depth == 0 or url in visited:
        return[url]
    visited.add(url)
    links = get_links(url)
    filtered_links = filter_links(links,base_domain)
    
    collected_links = [url]
    for link in filtered_links:
        absolute_url = urljoin(url,link)
        collected_links.extend(scrape_website(absolute_url,base_domain,depth-1,visited))    
    return list(set(collected_links))                

def scrape_urls(website,depth = 0):
    base_domain = urlparse(website).netloc.split('.')[0]
    links = scrape_website(website,base_domain,depth)
    return links()