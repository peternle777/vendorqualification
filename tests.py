import requests

def run_test(category, capabilities):
    print("\n=== Category: " + category + " ===")
    data = {"software_category": category, "capabilities": capabilities}
    print("Input:", data)

    resp = requests.post("http://localhost:8000/vendor_qualification", json=data)
    results = resp.json()["results"]

    print("Number of results:", len(results))
    for i, r in enumerate(results, 1):
        note = ""
        # Only annotate if the queried category is present in custom_categories
        if category.lower() in [c.lower() for c in r["custom_categories"]]:
            note = " (related to " + category + " according to description and category)"
        print(
            str(i) + ". " +
            r['product_name'] +
            " | Category: " + str(r['main_category']) +
            " | Custom Categories: " + ", ".join(r['custom_categories']) +
            " | Score: " + str(round(r['score'], 3)) +
            " | Similarity: " + str(round(r['similarity'], 3)) +
            note
        )

run_test("crm software", ["automation", "integration"])
run_test("accounting & finance software", ["budgeting"])
run_test("erp software", ["inventory"])
