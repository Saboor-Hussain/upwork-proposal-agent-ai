from urllib.parse import urlparse, urlunparse
import re

# Paste your long string of URLs here
raw_urls = """
https://insitechstaging.com/demo/pavilion/
https://insitechstaging.com/demo/pro-auto-towing/
https://insitechstaging.com/demo/stylist-factory/wp/
https://insitechstaging.com/demo/atf-contracting/
https://insitechstaging.com/demo/art-of-marbles
https://insitechstaging.com/demo/brochure-design/
https://insitechstaging.com/demo/financial-literacy
https://insitechstaging.com/demo/getting-sheared/wp-admin/
https://insitechstaging.com/demo/greatwork-logistics/
https://insitechstaging.com/demo/luckybackyard/customwp/
https://insitechstaging.com/demo/badboystrain/
https://insitechstaging.com/demo/barefoot-xpress
https://insitechstaging.com/demo/weapon-retention/
https://insitechstaging.com/demo/westminsterparty
https://insitechstaging.com/demo/boiling-wit/
https://insitechstaging.com/demo/bromley-collision
https://insitechstaging.com/demo/calgary-markets/wp-admin/
https://insitechstaging.com/demo/charles-clark
https://insitechstaging.com/demo/children-band/
https://insitechstaging.com/demo/cupsmx/
https://insitechstaging.com/demo/dakam/
https://insitechstaging.com/demo/DL-Tax-solution/
https://insitechstaging.com/demo/dropshipping/wp/
https://insitechstaging.com/demo/evergreen
https://insitechstaging.com/demo/lucky-old/
https://insitechstaging.com/demo/soldier-mountain-highland/
https://insitechstaging.com/demo/spectrum-group
https://insitechstaging.com/demo/sustainatech/
https://insitechstaging.com/demo/transportation-trend-setter/
https://insitechstaging.com/demo/jounrey/
https://insitechstaging.com/demo/kadobu
https://insitechstaging.com/demo/iron-gate
https://insitechstaging.com/demo/martialartsinthepark/
https://insitechstaging.com/demo/monsoon-security/
https://insitechstaging.com/demo/price-refractory/
https://insitechstaging.com/demo/stylist-factory/wp/wp-admin/
https://insitechstaging.com/demo/zee-media/
https://insitechstaging.com/demo/lazzat
https://insitechstaging.com/demo/philanthrocapitalistreview
https://insitechstaging.com/demo/vaultofbeauty
https://insitechstaging.com/demo/voxa-marketing/wp/
https://insitechstaging.com/demo/jeff-macgrandles/
https://insitechstaging.com/demo/contour-cabinets/
https://insitechstaging.com/demo/jonathan/
https://insitechstaging.com/demo/limitless-equine
https://insitechstaging.com/demo/weight-pullers/
https://insitechstaging.com/demo/conte-stanley/
https://insitechstaging.com/demo/eric-scott/
https://insitechstaging.com/demo/alberto-herrera/
https://insitechstaging.com/demo/kerry-kinard/
https://insitechstaging.com/demo/gary-yancy/
https://insitechstaging.com/demo/organx/
https://insitechstaging.com/demo/titans/
https://insitechstaging.com/demo/steve-grinding/
https://insitechstaging.com/demo/tnt-services/
https://insitechstaging.com/demo/rgc-groove/
https://insitechstaging.com/demo/circo-circus-funland/
https://insitechstaging.com/demo/four-sight/
https://insitechstaging.com/demo/rthompson/
https://insitechstaging.com/demo/brand-02
https://insitechstaging.com/demo/calender/
https://insitechstaging.com/demo/wyatt-johnson/
https://insitechstaging.com/demo/noir-moolah/
https://insitechstaging.com/demo/james-veon/
https://insitechstaging.com/demo/triple-star/
https://insitechstaging.com/demo/ferno/
https://insitechstaging.com/demo/zachary-farrow/
https://insitechstaging.com/demo/terra-nova/
https://insitechstaging.com/demo/AM-auto-body
https://insitechstaging.com/demo/travel-agency
https://insitechstaging.com/demo/relocate-travel
https://insitechstaging.com/demo/cruise-landing/
https://insitechstaging.com/demo/twb-company/
https://insitechstaging.com/our-case-studies/
"""

# Split the string into individual lines and clean each URL
url_set = set()
for line in raw_urls.splitlines():
    url = line.strip()
    if not url:
        continue
    # Remove fragments and query parameters
    parsed = urlparse(url)
    cleaned_path = re.sub(r'[#?].*$', '', parsed.path)
    if not cleaned_path.endswith('/'):
        cleaned_path += '/'
    normalized = urlunparse((parsed.scheme, parsed.netloc, cleaned_path, '', '', ''))
    url_set.add(normalized)

# Sort for readability
sorted_urls = sorted(url_set)

# Format as Python list
print("website_links = [")
for url in sorted_urls:
    print(f'    "{url}",')
print("]")
