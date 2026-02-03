#!/bin/bash
# Renumerar la clase 4 -> 3 en todos los splits (train, val, test)

for split in train val test; do
    echo "🔹 Procesando $split..."
    for f in $split/labels/*.txt; do
        # Reemplaza solo al inicio de la línea (clase)
        sed -i 's/^4 /3 /' "$f"
    done
done

echo "✅ Todas las etiquetas de clase 4 renumeradas a 3."
