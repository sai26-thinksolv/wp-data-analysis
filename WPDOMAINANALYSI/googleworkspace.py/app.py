import requests

def get_last_wp_entry(domain: str, content_type: str = "posts"):
    """
    Fetch last created and last modified WordPress post/page via REST API.

    Args:
        domain (str): Domain name (example.com).
        content_type (str): "posts" or "pages".

    Returns:
        dict with created and modified info, or None values if unavailable.
    """
    base_urls = [
        f"https://{domain}/wp-json/wp/v2/{content_type}",
        f"https://www.{domain}/wp-json/wp/v2/{content_type}"
    ]

    result = {
        "domain": domain,
        "type": content_type,
        "last_created": None,
        "last_modified": None,
        "status": "API not available"
    }

    for base_url in base_urls:
        try:
            # Get latest created (orderby=date)
            created_url = f"{base_url}?per_page=1&orderby=date&order=desc"
            r_created = requests.get(created_url, timeout=10)
            
            # Get latest modified (orderby=modified)
            modified_url = f"{base_url}?per_page=1&orderby=modified&order=desc"
            r_modified = requests.get(modified_url, timeout=10)

            if r_created.status_code == 200 and r_modified.status_code == 200:
                data_created = r_created.json()
                data_modified = r_modified.json()

                if isinstance(data_created, list) and len(data_created) > 0:
                    item = data_created[0]
                    result["last_created"] = {
                        "title": item.get("title", {}).get("rendered"),
                        "slug": item.get("slug"),
                        "date": item.get("date"),         # created date
                        "modified": item.get("modified"), # last updated
                        "link": item.get("link"),
                        "api_url": created_url
                    }

                if isinstance(data_modified, list) and len(data_modified) > 0:
                    item = data_modified[0]
                    result["last_modified"] = {
                        "title": item.get("title", {}).get("rendered"),
                        "slug": item.get("slug"),
                        "date": item.get("date"),         # created date
                        "modified": item.get("modified"), # last updated
                        "link": item.get("link"),
                        "api_url": modified_url
                    }

                result["status"] = "success"
                return result  # success, stop trying further URLs

        except Exception:
            continue  # try next base URL

    return result


def main():
    # 🔹 Add your sample domains here (without https or www)
    domains = [
        "jamesclear.com",
        "example.com",
        "wordpress.org"
    ]

    for d in domains:
        print(f"\nChecking domain: {d}")
        
        res_posts = get_last_wp_entry(d, content_type="posts")
        print("Posts:", res_posts)

        res_pages = get_last_wp_entry(d, content_type="pages")
        print("Pages:", res_pages)


if __name__ == "__main__":
    main()
