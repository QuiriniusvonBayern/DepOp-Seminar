# -------- Tests to compare word vectors ------------
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
        return "Keine Referenzwerte vorhanden.", False

    low, high = expected_ranges[(t1, t2)]

    if value < low:
        return f"➡ WARNUNG: Wert zu niedrig (erwartet: {low} bis {high})", False
    if value > high:
        return f"➡ WARNUNG: Wert zu hoch (erwartet: {low} bis {high})", False

    return f"✔ OK (innerhalb erwarteter Range {low}–{high})", True

def interpret_neighbors(token, neighbors, test, test_range):
    comments = []
    success = True

    # -----------------------------
    # 1. Gruppierungslogik nach Präfix
    # -----------------------------
    if test != "none" and token.startswith(f"{test}_"):

        expected_prefix = f"{test}_"
        subset = neighbors[:min(test_range, len(neighbors))]

        all_correct = True
        for w, _ in subset:
            if not w.startswith(expected_prefix):
                comments.append(
                    f"➡ WARNUNG: Nachbar {w} entspricht nicht der erwarteten Gruppe ({test})."
                )
                all_correct = False
                success = False

        if all_correct:
            comments.append("✔ OK: Token gruppieren sich erwartungsgemäß.")
    return comments, success



# ---------------------------------------------------
# Semantische Ähnlichkeit prüfen
# ---------------------------------------------------

def check_token_similarity(model, token_pairs, verbose):
    if verbose:
        print("\n============================================")
        print(" SEMANTISCHE ÄHNLICHKEITEN (COSINE CHECK)")
        print("============================================")

    success_counter = 0

    for t1, t2 in token_pairs:
        try:
            sim = model.wv.similarity(t1, t2)
            

            judgement, ok = interpret_similarity((t1, t2), sim)
            if verbose:
                print(f"\n{t1} ↔ {t2} = {sim:.4f}")
                print("Beurteilung:", judgement)

            if ok:
                success_counter += 1

        except KeyError:
            print(f"{t1} oder {t2} nicht im Vokabular")

    return success_counter



# ---------------------------------------------------
# Nachbaranalyse
# ---------------------------------------------------

def check_neighbors(model, tokens, test, test_range, verbose, topn=10):
    if verbose:
        print("\n============================================")
        print(" NEAREST-NEIGHBOR CHECK")
        print(f" {test.upper()}")
        print("============================================")

    success_counter = 0

    for t in tokens:
        if verbose:
            print(f"\nTop-{topn} Nachbarn für {t}:")
        try:
            nn = model.wv.most_similar(t, topn=topn)
            if verbose:
                for w, s in nn:
                    print(f"  {w:30s} {s:.4f}")

            comments, ok = interpret_neighbors(t, nn, test, test_range)

            if ok:
                success_counter += 1
            if verbose:
                for c in comments:
                    print("Beurteilung:", c)

        except KeyError:
            print(f"{t} nicht im Vokabular")

    return f"{success_counter} von {len(tokens)} Erfolgreich"

# ---------------------------------------------------
# Gesamter Qualitätscheck
# ---------------------------------------------------

def run_full_quality_check(model_list, verbose):
    print("\n################################################")
    print("        AUTOMATISCHER W2V QUALITY CHECK")
    print("               Model Parameters:       ")
    print(f"Vektorraumgröße = {model_list[1]['Vector Size']} Fenstergröße  = {model_list[1]['Window']} Algorithmus = {model_list[1]['Algorithmus']}")
    print(f"         Epochen = {model_list[1]['Epochs']} Anzahl der Sätze = {model_list[1]['Satzanzahl']} Länge der Sätze = {model_list[1]['Satzlänge']}")
    print("################################################\n")

    model = model_list[0]
    success = {}

    # 1. Semantische Erwartungspaare
    success_counter = check_token_similarity(model, [
        ("Gender_Female", "Gender_Male"),
        ("Credit_Score_Fair", "Credit_Score_Good"),
        ("Churn_Yes", "Churn_No"),
        ("Country_France", "Country_Germany"),
        ("Country_France", "Country_Spain"),
    ], verbose=verbose)
    success["Semantische Erwartungspaare: "] = f"{success_counter} von 5 Erfolgreich"

    # 2. Länder-Nachbarn
    success["Länder-Nachbarn: "] = check_neighbors(
        model, 
        ["Country_France", "Country_Germany", "Country_Spain"], 
        test="Country", 
        test_range=2, 
        verbose=verbose
    )

    # 3. Alters-Nachbarn
    success["Alters-Nachbarn: "] = check_neighbors(
        model,
        ["Age_Young_Adults", "Age_Adults_in_their_Prime", "Age_Middle_aged"],
        test="Age",
        test_range=4, 
        verbose=verbose
    )

    # 4. Gehalts-Nachbarn
    success["Gehalts-Nachbarn: "] = check_neighbors(
        model,
        ["Salary_Very_low", "Salary_Below_average", "Salary_Very_high"],
        test="Salary",
        test_range=6, 
        verbose=verbose
    )

    # 5. Credit_Score-Nachbarn
    success["Credit_Score-Nachbarn: "] = check_neighbors(
        model,
        ["Credit_Score_Poor", "Credit_Score_Good", "Credit_Score_Excellent"],
        test="Credit_Score",
        test_range=4, 
        verbose=verbose
    )

    # 6. Churn-Nachbarn
    success["Churn-Nachbarn: "] = check_neighbors(
        model,
        ["Churn_Yes", "Churn_No"],
        test="Churn",
        test_range=1, 
        verbose=verbose
    )

    # Ausgabe
    for x, y in success.items():
        print(x, y)

    return success
