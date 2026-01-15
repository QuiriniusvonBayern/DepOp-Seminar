# basic_vector_tests.py
# ---------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------
def interpret_similarity(token_pair, value):
    t1, t2 = token_pair
    expected_ranges = {
        ("Gender_Female", "Gender_Male"): (0.30, 0.90),
        ("Credit_Score_Fair", "Credit_Score_Good"): (0.25, 0.85),
        ("Churn_Yes", "Churn_No"): (-0.20, 0.30),
        ("Country_France", "Country_Germany"): (0.20, 0.80),
        ("Country_France", "Country_Spain"): (0.20, 0.80),
    }

    if (t1, t2) not in expected_ranges:
        return "Keine Referenz", False

    low, high = expected_ranges[(t1, t2)]
    if value < low: return "Zu niedrig", False
    if value > high: return "Zu hoch", False
    return "OK", True

def interpret_neighbors(token, neighbors, test, test_range):
    # Gibt nun direkt die Anzahl der korrekten Matches zurück
    if test != "none" and token.startswith(f"{test}_"):
        expected_prefix = f"{test}_"
        subset = neighbors[:min(test_range, len(neighbors))]
        
        matches = sum(1 for w, _ in subset if w.startswith(expected_prefix))
        total_checked = len(subset)
        
        return matches, total_checked
    return 0, 0

# ---------------------------------------------------
# Semantische Ähnlichkeit prüfen
# ---------------------------------------------------
def check_token_similarity(model, token_pairs, verbose):
    results = []
    
    for t1, t2 in token_pairs:
        try:
            sim = float(model.wv.similarity(t1, t2))
            msg, passed = interpret_similarity((t1, t2), sim)
            
            results.append({
                "pair": f"{t1}-{t2}",
                "similarity": sim,
                "passed": passed,
                "msg": msg
            })
            
            if verbose:
                print(f"{t1} ↔ {t2} = {sim:.4f} ({msg})")

        except KeyError:
            if verbose: print(f"{t1} oder {t2} fehlen.")
            results.append({
                "pair": f"{t1}-{t2}",
                "similarity": None,
                "passed": False,
                "msg": "Missing Token"
            })

    return results

# ---------------------------------------------------
# Nachbaranalyse
# ---------------------------------------------------
def check_neighbors(model, tokens, test, test_range, verbose, topn=10):
    results = []
    
    for t in tokens:
        try:
            nn = model.wv.most_similar(t, topn=topn)
            matches, checked = interpret_neighbors(t, nn, test, test_range)
            
            # Wir speichern die Quote (z.B. 1.0 = 100% korrekt)
            score = matches / checked if checked > 0 else 0
            
            results.append({
                "token": t,
                "category": test,
                "matches": matches,
                "checked_range": checked,
                "score": score
            })
            
            if verbose:
                print(f"{t}: {matches}/{checked} Nachbarn korrekt.")

        except KeyError:
            if verbose: print(f"{t} nicht gefunden.")
    
    return results

# ---------------------------------------------------
# Gesamter Qualitätscheck
# ---------------------------------------------------
def run_full_quality_check(model_list, verbose):
    if verbose:
        print("\n################################################")
        print("        AUTOMATISCHER W2V QUALITY CHECK")
        print("               Model Parameters:       ")
        print(f"Vektorraumgröße = {model_list[1]['Vector Size']} Fenstergröße  = {model_list[1]['Window']} Algorithmus = {model_list[1]['Algorithmus']}")
        print(f"         Epochen = {model_list[1]['Epochs']} Anzahl der Sätze = {model_list[1]['Satzanzahl']} Länge der Sätze = {model_list[1]['Satzlänge']}")
        print("################################################\n")

    if verbose:
        print("Running Basic Quality Checks...")

    model = model_list[0]
    # Wir sammeln alles in einem Dictionary
    data = {
        "semantic_similarity": [],
        "neighbors": []
    }

    # 1. Semantische Erwartungspaare
    data["semantic_similarity"] = check_token_similarity(model, [
        ("Gender_Female", "Gender_Male"),
        ("Credit_Score_Fair", "Credit_Score_Good"),
        ("Churn_Yes", "Churn_No"),
        ("Country_France", "Country_Germany"),
        ("Country_France", "Country_Spain"),
    ], verbose=verbose)

    # 2. Nachbar-Tests (Sammeln in einer Liste)
    neighbor_configs = [
        (["Country_France", "Country_Germany", "Country_Spain"], "Country", 2),
        (["Age_Young_Adults", "Age_Adults_their_Prime", "Age_Middle_aged"], "Age", 4),
        (["Salary_Very_low", "Salary_Below_average", "Salary_Very_high"], "Salary", 6),
        (["Credit_Score_Poor", "Credit_Score_Good", "Credit_Score_Excellent"], "Credit_Score", 4),
        (["Churn_Yes", "Churn_No"], "Churn", 1)
    ]

    for tokens, category, rng in neighbor_configs:
        res = check_neighbors(model, tokens, test=category, test_range=rng, verbose=verbose)
        data["neighbors"].extend(res)

    return data
