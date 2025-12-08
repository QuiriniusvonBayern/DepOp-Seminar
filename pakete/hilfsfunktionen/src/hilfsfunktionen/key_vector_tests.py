# ------- Deinition of Tests to compare Key-Vectors -----------
def get_key_values(key, filter, prefix="key", df_with_keys=None):
    values_with_keys = df_with_keys.iloc[key] #mit Keys
    if filter:
        return [w for w in values_with_keys if not w.startswith(prefix)]
    else:
        return [w for w in values_with_keys]

def compare_lists(list1, list2):
    output:str = ""
    anzahl_paare:int = 0
    gleiche_paare = [(i, a, b) for i, (a, b) in enumerate(zip(list1, list2)) if a == b]
    anzahl_paare = len(gleiche_paare)
    output += f"{anzahl_paare} von {len(list1)} Gleiche Werte: "
    for pos, a, b in gleiche_paare:
        output += f"(Position {pos} : '{a}' )"
    return output, anzahl_paare



# ---------------------------------------------------
# Key-Strukturprüfung
# ---------------------------------------------------

def check_key_structure(model, keys, verbose, df_with_keys):
    if verbose:
        print("\n============================================")
        print(" KEY-STRUKTUR CHECK")
        print("============================================")
    successfull_pairs = ""
    i :int =0
    for k in keys:
        successfull_pairs += f"Analysis for {k} \n"
        if verbose:
            print(f"\nWerte in Zeile {k}:")
            print(get_key_values(int(k.replace("key_", "")), True, df_with_keys=df_with_keys))
            print(f"\nNachbarn für {k}:")

        try:
            nn = model.wv.most_similar(k, topn=5)
            idx = 0
            for w, s in nn:
                comments, num_pairs = compare_lists(get_key_values(int(w.replace("key_", "")), True, df_with_keys=df_with_keys) , get_key_values(int(k.replace("key_", "")), True, df_with_keys=df_with_keys))
                successfull_pairs += f"Matching pairs in the {idx}th most similar: ({num_pairs}/11) \n"
                idx += 1
                if verbose:
                    print(f"  {w:30s} {s:.4f}")
                    print(f"\nWerte in Zeile {k}:")
                    print(get_key_values(int(w.replace("key_", "")), True))
           
            if verbose:           
                print("Beurteilung:", comments)

        except KeyError:
            print(f"{k} nicht im Vokabular")
        i += 1

    return successfull_pairs

def run_full_quality_check_key(model_list, verbose, df_with_keys=None):
    print("\n################################################")
    print("        AUTOMATISCHER W2V KEY CHECK")
    print("################################################\n")

    model = model_list[0]
    success = {}

    # 7. Key-Struktur
    key_ok = check_key_structure(model, ["key_1", "key_2", "key_3"], verbose=verbose, df_with_keys=df_with_keys)
    success[f"Key-Struktur: \n"] = key_ok

    # Ausgabe
    for x, y in success.items():
        print(x, y)
    return success