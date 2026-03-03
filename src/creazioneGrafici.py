import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- DATI DELL'ESPERIMENTO ---

data = {
    '% Missing':       [0, 10,  20,  30,  40,  50,  60,  70,  80,  90],
    'Acc_Senza':       [82.4, 77.6, 72.8, 68.4, 66.1, 65.0, 60.5, 57.7, 58.5, 56.3],
    'Sens_Senza':      [88.9, 87.4, 85.3, 83.2, 86.8, 89.5, 88.9, 95.3, 97.4, 97.9],
    'Spec_Senza':      [74.9, 66.5, 58.7, 51.5, 42.5, 37.1, 28.1, 15.0, 14.4, 9.0],
    'Acc_Con':         [82.4, 80.7, 81.0, 77.9, 73.1, 75.4, 71.1, 69.2, 63.9, 57.1],
    'Sens_Con':        [88.9, 85.8, 79.5, 78.9, 73.7, 81.6, 78.9, 76.8, 73.7, 82.1],
    'Spec_Con':        [74.9, 74.9, 82.6, 76.6, 72.5, 68.3, 62.3, 60.5, 52.7, 28.7]
}
df = pd.DataFrame(data)

# Imposta stile grafico
sns.set_theme(style="whitegrid")

# --- GRAFICO 1: ACCURATEZZA ---

print("Generazione grafico Accuratezza...")
plt.figure(figsize=(10, 6))
sns.lineplot(x='% Missing', y='Acc_Senza', data=df, marker='o', label='Senza Split Surrogati', color='red')
sns.lineplot(x='% Missing', y='Acc_Con', data=df, marker='o', label='Con Split Surrogati', color='blue')

# Personalizzazione
plt.title("Confronto dell'Accuratezza al Variare dei Dati Mancanti", fontsize=16)
plt.xlabel("Percentuale di Dati Mancanti (%)", fontsize=12)
plt.ylabel("Accuratezza (%)", fontsize=12)
plt.ylim(50, 90) # Limiti dell'asse Y per una migliore visualizzazione
plt.xlim(-5, 95) # Aggiunge un po' di margine all'asse X
plt.legend()

# Salva il file
plt.savefig("grafico_accuratezza.pdf", format='pdf', bbox_inches='tight')
print("...salvato in 'grafico_accuratezza.pdf'")
plt.close() # Chiude la figura per liberare memoria

# --- GRAFICO 2: SENSIBILITÀ ---

print("Generazione grafico Sensibilità...")
plt.figure(figsize=(10, 6))
sns.lineplot(x='% Missing', y='Sens_Senza', data=df, marker='o', label='Senza Split Surrogati', color='red')
sns.lineplot(x='% Missing', y='Sens_Con', data=df, marker='o', label='Con Split Surrogati', color='blue')

# Personalizzazione
plt.title("Confronto della Sensibilità al Variare dei Dati Mancanti", fontsize=16)
plt.xlabel("Percentuale di Dati Mancanti (%)", fontsize=12)
plt.ylabel("Sensibilità (%)", fontsize=12)
plt.ylim(70, 100)
plt.xlim(-5, 95)
plt.legend()

# Salva il file
plt.savefig("grafico_sensibilita.pdf", format='pdf', bbox_inches='tight')
print("...salvato in 'grafico_sensibilita.pdf'")
plt.close()

# --- GRAFICO 3: SPECIFICITÀ ---

print("Generazione grafico Specificità...")
plt.figure(figsize=(10, 6))
sns.lineplot(x='% Missing', y='Spec_Senza', data=df, marker='o', label='Senza Split Surrogati', color='red')
sns.lineplot(x='% Missing', y='Spec_Con', data=df, marker='o', label='Con Split Surrogati', color='blue')

# Personalizzazione
plt.title("Confronto della Specificità al Variare dei Dati Mancanti", fontsize=16)
plt.xlabel("Percentuale di Dati Mancanti (%)", fontsize=12)
plt.ylabel("Specificità (%)", fontsize=12)
plt.ylim(0, 90)
plt.xlim(-5, 95)
plt.legend()

# Salva il file
plt.savefig("grafico_specificita.pdf", format='pdf', bbox_inches='tight')
print("...salvato in 'grafico_specificita.pdf'")
plt.close()

print("\nOperazione completata. Tutti i grafici sono stati generati.")