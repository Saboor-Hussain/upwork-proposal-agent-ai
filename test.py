import csv
from urllib.parse import urlparse

# Change this to your actual file path if needed
csv_file = './data/working-websites.csv'
muhammadprojects_file = './data/muhammadprojects_websites.csv'
insitechstaging_file = './data/insitechstaging_websites.csv'
other_file = './data/other_websites.csv'

muhammadprojects = []
insitechstaging = []
other = []

# Helper to extract domain (without www)
def extract_domain(url):
    try:
        parsed = urlparse(url if url.startswith('http') else 'http://' + url)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception:
        return ''

# Read the CSV and separate URLs by domain
with open(csv_file, newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        if row and row[0].strip():
            url = row[0].strip()
            domain = extract_domain(url)
            if domain.endswith('muhammadprojects.com'):
                muhammadprojects.append([url])
            elif domain.endswith('insitechstaging.com'):
                insitechstaging.append([url])
            else:
                other.append([url])

# Write each group to its own CSV file
def write_csv(filename, rows):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Website Link'])
        writer.writerows(rows)

write_csv(muhammadprojects_file, muhammadprojects)
write_csv(insitechstaging_file, insitechstaging)
write_csv(other_file, other)

print(f"Saved {len(muhammadprojects)} muhammadprojects.com websites to {muhammadprojects_file}")
print(f"Saved {len(insitechstaging)} insitechstaging.com websites to {insitechstaging_file}")
print(f"Saved {len(other)} other websites to {other_file}")
