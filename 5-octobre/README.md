# Tableau de Bord d'Analyse E-commerce

Ce tableau de bord fournit des informations sur les indicateurs de performance e-commerce et l'analyse du comportement client.

## Explication des Métriques Clés

### Métriques de Vente

**Panier Moyen (PM)**
- Définition : Le montant moyen dépensé par commande finalisée
- Calcul : Chiffre d'affaires total / Nombre de commandes finalisées
- Signification : Tendances des dépenses clients et efficacité de la stratégie tarifaire

**Taux de Conversion**
- Définition : Pourcentage de paniers convertis en commandes finalisées
- Calcul : (Commandes finalisées / Total des paniers) × 100
- Signification : Efficacité du processus de paiement et de l'expérience utilisateur globale

**Taux d'Abandon de Panier**
- Définition : Pourcentage de paniers abandonnés
- Calcul : (Paniers abandonnés / Total des paniers) × 100
- Signification : Problèmes potentiels de tarification, processus de paiement ou expérience utilisateur

### Métriques de Comportement Client

**Abandon de Panier par Valeur**
- Contenu : Distribution des valeurs des paniers abandonnés
- Importance : Aide à identifier les seuils de prix où les clients sont plus susceptibles d'abandonner

**Distribution Géographique**
- Contenu : Répartition des commandes par pays
- Importance : Aide à comprendre la performance des marchés internationaux et les modèles d'expédition

**Distribution des Moyens de Paiement**
- Contenu : Méthodes de paiement préférées
- Importance : Aide à optimiser les options de paiement et réduire les frictions

## Questions Fréquentes

### Questions Générales

**Q : Qu'est-ce qu'un panier abandonné ?**
R : Un panier est considéré comme abandonné lorsque des articles sont ajoutés mais que le processus de paiement n'est pas finalisé dans les 24 heures.

**Q : Comment le chiffre d'affaires est-il calculé ?**
R : Le chiffre d'affaires est calculé uniquement sur les commandes finalisées, excluant les commandes annulées ou remboursées.

### Questions sur les Données

**Q : À quelle fréquence les données sont-elles mises à jour ?**
R : Les données sont mises à jour en temps réel pour les nouvelles commandes et l'activité des paniers.

**Q : Pourquoi peut-il y avoir des écarts dans les valeurs monétaires ?**
R : Les commandes peuvent être passées dans différentes devises (EUR, USD). Toutes les métriques sont normalisées en EUR pour la cohérence.

### Questions Techniques

**Q : Comment la saisonnalité est-elle prise en compte dans les métriques ?**
R : Les tendances sont analysées sur une base annuelle et mensuelle pour tenir compte des variations saisonnières.

**Q : Comment les valeurs aberrantes sont-elles traitées ?**
R : Les valeurs extrêmes (ex : commandes inhabituellement importantes) sont signalées mais incluses dans les calculs sauf si manifestement erronées.

## Dictionnaire des Données

### Statut de Commande
- `Livré` : Commande livrée
- `Annulée` : Commande annulée
- `Remboursé` : Commande remboursée
- `Panier abandonné` : Panier non finalisé

### Moyens de Paiement
- `Card via Stripe` : Paiement par carte bancaire
- `Alma - Paiement en 3 fois` : Paiement en trois fois
- `Transfert bancaire` : Virement bancaire

### Modes de Livraison
- `DHL` : Livraison standard DHL
- `CLICK AND COLLECT` : Retrait en magasin

## Bonnes Pratiques d'Analyse

1. **Comparer des Périodes Similaires**
   - Utiliser des comparaisons d'une année sur l'autre quand possible
   - Tenir compte des variations saisonnières

2. **Considérer Plusieurs Métriques**
   - Examiner les métriques connexes ensemble
   - Exemple : Un panier moyen élevé mais un faible taux de conversion peut indiquer des problèmes de tarification

3. **Le Contexte est Important**
   - Prendre en compte les facteurs externes (promotions, jours fériés)
   - Rechercher des modèles dans le comportement client

## Support

Pour des questions supplémentaires ou un support technique, veuillez contacter l'équipe d'analyse.

## Contribution

Pour suggérer des améliorations ou signaler des problèmes :
1. Ouvrir une issue décrivant l'amélioration/problème
2. Fournir des exemples spécifiques si possible
3. Inclure des échantillons de données pertinents si applicable

## Chatbot Integration

This dashboard includes a natural language chatbot powered by the Mistral LLM. Users can ask questions about sales, revenue trends, CLV, RFM segments, and more.

### Features

- **Natural Language Queries:** Ask questions in plain English and receive insightful answers.
- **Retrieval-Augmented Generation (RAG):** When enabled, the chatbot retrieves relevant documents to provide accurate and context-aware responses.
- **Interactive Chat Interface:** Seamlessly integrated within the Streamlit dashboard.

### How to Use

1. Navigate to the **Ask a Question** section in the dashboard.
2. Enter your query in the input box.
3. Click **Get Insights** to receive an answer.
4. Optionally, enable **Retrieval (RAG)** to include additional context from your data.

### Fine-Tuning the Model

To fine-tune the Mistral model with your domain-specific data:

1. Ensure you have the necessary GPU resources.
2. Prepare your QA pairs in `training_data.jsonl`.
3. Run the fine-tuning script using Hugging Face's transformers library with PEFT/LoRA.
4. Update the `MistralClient` with the new model checkpoint.

### Contributing

For contributions, please refer to the [Contribution Guidelines](CONTRIBUTING.md).
