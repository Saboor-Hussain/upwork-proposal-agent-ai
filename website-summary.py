import csv
import requests
import os
import time

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDez_YRD1X6659wsA9VPAeBUcb49vYQtOw")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY

INPUT_CSV = './data/working-websites2.csv'
OUTPUT_CSV = './website-data2.csv'

# CATEGORY_CONTEXT = """
# E-commerce: Online stores, product catalogs, shopping carts, B2B/B2C marketplaces, dropshipping platforms, auction sites, digital product stores
# Portfolio: Personal portfolios, artist galleries, freelancer showcases, creative resumes, photographer portfolios, designer portfolios
# Blog: News, articles, personal blogs, niche blogs, magazine sites, opinion/editorial blogs
# Corporate: Company websites, business landing pages, consulting firms, law firms, financial services, agency sites
# Educational: Online courses, school/university sites, e-learning platforms, training centers, tutoring services, educational resources
# Nonprofit: Charity organizations, NGOs, fundraising sites, advocacy groups, community organizations
# Healthcare: Hospitals, clinics, medical practices, health information portals, telemedicine, wellness centers
# Real Estate: Property listings, realtor sites, rental platforms, real estate agencies, property management
# Entertainment: Music bands, movie sites, event promotion, streaming platforms, fan sites, gaming communities
# Travel & Hospitality: Hotel booking, travel agencies, tour operators, travel blogs, airline sites, vacation rentals
# Technology: SaaS products, software companies, app landing pages, tech startups, IT services
# Food & Restaurant: Restaurant menus, food delivery, catering services, recipe blogs, cafes, bars
# Marketplace: Multi-vendor platforms, service marketplaces, gig economy sites, classified ads
# Finance: Banking, investment, insurance, fintech, personal finance blogs, loan services
# Government: Official government portals, city/municipality sites, public service information
# Sports & Fitness: Sports teams, gyms, fitness trainers, sports news, event registration, fan clubs
# Automotive: Car dealerships, auto repair, car rental, automotive blogs, parts stores
# Fashion & Beauty: Fashion brands, beauty salons, online boutiques, style blogs, cosmetics stores
# Media & Publishing: News outlets, magazines, book publishers, digital media, podcast sites
# Construction & Industrial: Construction companies, engineering firms, industrial suppliers, manufacturing
# Wedding & Events: Wedding planners, event venues, invitation sites, event management
# Children & Family: Daycares, parenting blogs, kids' activities, family services
# Pets & Animals: Veterinary clinics, pet stores, animal shelters, pet care blogs
# Other: Any website that does not fit the above categories
# """

CATEGORY_CONTEXT = """
Health & Wellness: PersonalTraining, Yoga, Nutritionists, MentalHealth, BodyandMindRebalance, FitnessTraining, Boxing, PhysicalGaming, HealthPerformance, LifestyleHealth
Medical Services: Dentist, Therapist, Consultant, MassageSpecialist, MedicalGroupClinic(In-HouseLab, VirtualConsultation, DiabetesClinic), Medicine&InfectiousDiseaseServices, BackPain, Orthopedic, SpinalCord, CovidRapidTest(TestKit, Self-Test), DentalITSupport
Real Estate: PropertyManagement, BuyandSell, Investments, PropertyListing, PricingCalculator, HomeBuyingandSelling, RealEstateDevelopment(Construction, Financing, Operations, Management), BusinessBuy&Sell, BusinessBroker
E-commerce: CampingGears, BarberProducts, Jewelry(Earrings, Piercing), Apparel, GymWear, AthleticWear, Books, OnlineFoodStore, Cakes&Desserts(OrderOnline), Salon&SPA, MedicalSupplements, WooCommerce
Solar Energy: SolarPanels, SolarSolutions, AuditEnergy, SolarInstallation, RequestaQuote
Business Services: BusinessConsultancy, BusinessAdvisor, FinancialBlog, BusinessSolutions, ITConsultant, ProfessionalDevelopment, TaxAdvisoryforDentists, BusinessFundings, FinancialFreedom, BusinessLoans
Financial Services: Loan, CapitalSolutions, FarmerFinancing&Insurance, HomeLoan, Mortgage, MortgageCalculator, InvestmentCompany, Crypto(MoneyMarket, DigitalCurrency), VentureCapital
Legal Services: LawFirm, DivorceLawyer, InjuryLawyer(AutoInjury), AccidentalCoverage, Settlements, LegalConsultation
Digital Agency: DigitalMarketing, SEO, PPC, YouTubeMonetization, ContentMonetization
Transportation & Logistics: RentACar, LuxuryTransport, CourierServices, FreightServices, Logistics, VehicleRental(Vans&SUVs), AutoTowing, MaritimeLogistics(ShippingAgency, ShipContainers)
Cleaning Services: GutterCleaning, DuctCleaning, WindowCleaning, SewerageCleaning, CarWash, GeneralCleaning
Construction & Home Services: Construction, StairsInstallation, MetalFabrication, Roofing, Plumbing, ElectricalSolutions, HomeAutomation, OfficeAutomation, HomeStaging, HomeDecor
Education & Learning: LanguageEducation(Bilingual, GiftingFunctionality), LearningPlatform(App, Webinars), CosmetologyInstitute(Skincare, SalonCourses), Schooling, LearningVariances(Dyslexia, ADHD, Speech)
Non-Profit & NGO: MentalHealthAwareness, EducationFunding, SoldierSupport, Donations, SeniorCare, JobCreation, DogNGO
Travel & Tourism: VacationServices, Tour&Tourism, OnlineBooking, VacationRentals
Creative & Media: Photography, Creagraphy, ArtTeaching, MusicStudios, MusicArtists, ScreenPrinting, Embroidery, CustomPrinting, FashionArticles
Technology & IT: DataOrganization, AIChatBot, PersonalAssistant, CloudComputing, TechServices, VPN, VPS, ITSolutions
Gaming: OnlineGamingCoaching, FitnessGaming(Boxing, Physical)
Food & Hospitality: Restaurant(Booking, Catering), ItalianKitchen(Order, Reservation), Steaks, CateringServices
Pet Care: PetSitting, DogCare, AnimalCare(Signup)
Manufacturing & Industrial: Manufacturing(Materials, Scaffolding), EnvironmentalIndustrial, Engineering
Retail & Specialty: VendingMachines, JewelryStores
Event Planning: WeddingPlanning, EventManagement
Career & Recruitment: RecruitmentAgency, ResumeServices, CareerCounseling, Coaching
Sports & Recreation: TennisCourts, Baseball
Aviation: AviationCompany
Space & Science: SpaceWeatherAnalysis, SpaceResearch
Fashion & Beauty: FashionEthics, Salon&SPA, Cosmetology
Online Services: SocialNetworkLandingPage, Translation, Proofreading, Transcription, ContentRemovalServices
NFT & Blockchain: NFT, NFTTrading
Society & Community: PalmTreeSociety, CommunityPropertyManagement
Marketing & Advertising: AmazonCampaigns, LeadGeneration, HubSpot
Other: Assign the Category according to the website's main focus or service.
"""


def get_website_summary(url):
    prompt = f"""
        Visit the website: {url}
        1. Identify the main purpose or goal of the website.
        2. Suggest the most relevant category for this website from the following list:\n{CATEGORY_CONTEXT}
        3. If the website does not fit any of the above categories, assign a new category according to the website's main focus or service.
        4. Write a short summary (2-3 sentences) describing the website.
        5. List 3-5 relevant keywords for the website.
        6. Identify the main tech stack or platform used to build the website. Only output the tech stack as a comma-separated list (e.g., WordPress, Elementor). Do not explain or justify your choice.
        Format your response as:
        Category: <category>\nDescription: <summary>\nKeywords: <comma-separated keywords>\nTechstacks: <comma-separated techstacks>\n

    """
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            # Parse the response
            category = description = keywords = techstacks = ""
            for line in text.splitlines():
                if line.lower().startswith("category:"):
                    category = line.split(":", 1)[1].strip()
                elif line.lower().startswith("description:"):
                    description = line.split(":", 1)[1].strip()
                elif line.lower().startswith("keywords:"):
                    keywords = line.split(":", 1)[1].strip()
                elif line.lower().startswith("techstacks:"):
                    techstacks = line.split(":", 1)[1].strip()
            return category, description, keywords, techstacks
        else:
            return "Error", f"API Error: {response.status_code}", "", ""
    except Exception as e:
        return "Error", f"Exception: {str(e)}", "", ""

def main():
    results = []
    with open(INPUT_CSV, 'r', newline='', encoding='utf-8') as infile:
        reader = list(csv.reader(infile))
        total = len([row for row in reader if row and row[0].strip()])
        infile.seek(0)
        reader = csv.reader(infile)
        idx = 0
        for row in reader:
            if not row or not row[0].strip():
                continue
            idx += 1
            url = row[0].strip()
            print(f"Processing {idx} out of {total}: {url}")
            category, description, keywords, techstacks = get_website_summary(url)
            result = {
                'Website Links': url,
                'Category': category,
                'Description': description,
                'Keywords': keywords,
                'Techstacks': techstacks
            }
            results.append(result)
            # Write/update after each result
            with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=['Website Links', 'Category', 'Description', 'Keywords', 'Techstacks'])
                writer.writeheader()
                writer.writerows(results)
            time.sleep(30)  
if __name__ == '__main__':
    main()
