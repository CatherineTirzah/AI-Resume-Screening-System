import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load job description
with open("job_description.txt", "r") as file:
    job_description = file.read()

# Load resumes
resume_folder = "resumes"
resumes = []
resume_names = []

for file in os.listdir(resume_folder):
    if file.endswith(".txt"):
        with open(os.path.join(resume_folder, file), "r") as f:
            resumes.append(f.read())
            resume_names.append(file)

# Combine job description with resumes
documents = [job_description] + resumes

# Convert text to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate similarity
similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Rank resumes
ranked_resumes = sorted(zip(resume_names, similarity_scores), key=lambda x: x[1], reverse=True)

print("\nResume Ranking:\n")

for rank, (name, score) in enumerate(ranked_resumes, start=1):
    print(f"{rank}. {name} - Similarity Score: {score:.2f}")