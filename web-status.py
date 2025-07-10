import csv
import requests

INPUT_CSV = './data/all-websites.csv'
OUTPUT_CSV = './data/websites-status.csv'


def check_website_status(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return 'Working'
        else:
            return f'Error {response.status_code}'
    except Exception as e:
        return f'Not Working ({str(e)})'


def main():
    results = []
    with open(INPUT_CSV, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if not row or not row[0].strip():
                continue
            url = row[0].strip()
            status = check_website_status(url)
            print(f"Checked: {url} => {status}")  # Print status to terminal
            results.append({'Links': url, 'Status': status})

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['Links', 'Status'])
        writer.writeheader()
        writer.writerows(results)


if __name__ == '__main__':
    main()
