from model import generate_proposal

# Example test jobs
TEST_JOBS = [
    {
        "title": "Build a Responsive Portfolio Website",
        "description": "I need a modern, responsive portfolio website to showcase my work. It should be built with React and have a contact form.",
        "category": "Web Development"
    },
    {
        "title": "Design a Logo for a Tech Startup",
        "description": "Looking for a creative logo designer to create a unique logo for our new tech startup. Experience with branding is a plus.",
        "category": "Graphics Designing"
    }
]

def main():
    for job in TEST_JOBS:
        print(f"\n--- Proposal for: {job['title']} ---\n")
        proposal = generate_proposal(job['title'], job['description'], job['category'])
        print(proposal)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
