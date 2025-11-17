#!/usr/bin/env python3
"""
Power User Message Analysis

Extracts message samples from top 10 tutors and top 10 students
to verify they are human-written pedagogical content, not bots.

Usage:
    python scripts/analyze_power_users.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data" / "raw"

    input_file = data_dir / "mathconverse_full_dataset.parquet"

    if not input_file.exists():
        print(f"ERROR: {input_file} not found")
        return 1

    print(f"Loading dataset from {input_file.name}...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} messages\n")

    # Convert binary indicators to role
    df['student_binary'] = pd.to_numeric(df['student'], errors='coerce').fillna(0)
    df['helper_binary'] = pd.to_numeric(df['helper'], errors='coerce').fillna(0)

    df['role'] = 'unknown'
    df.loc[df['student_binary'] == 1, 'role'] = 'student'
    df.loc[df['helper_binary'] == 1, 'role'] = 'tutor'

    # Get top 10 tutors by message count
    print("="*80)
    print("TOP 10 TUTORS BY MESSAGE COUNT")
    print("="*80 + "\n")

    tutor_df = df[df['role'] == 'tutor']
    top_tutors = tutor_df.groupby('author_id').size().nlargest(10)

    for rank, (tutor_id, msg_count) in enumerate(top_tutors.items(), 1):
        tutor_msgs = tutor_df[tutor_df['author_id'] == tutor_id]

        # Get tutor name (from first message)
        tutor_name = tutor_msgs.iloc[0]['author_name'] if 'author_name' in tutor_msgs.columns else f"Tutor_{tutor_id}"

        # Get conversation count
        conv_count = tutor_msgs['conversation_id'].nunique()

        # Sample 5 random messages
        sample_size = min(5, len(tutor_msgs))
        samples = tutor_msgs.sample(n=sample_size, random_state=42)

        print(f"#{rank}. {tutor_name} (ID: {tutor_id})")
        print(f"    {msg_count:,} messages across {conv_count:,} conversations")
        print(f"    Sample messages:\n")

        for i, (_, msg) in enumerate(samples.iterrows(), 1):
            text = msg['content'] if 'content' in msg else msg.get('text', '[no text field]')
            # Truncate long messages
            if len(text) > 150:
                text = text[:150] + "..."
            print(f"      [{i}] {text}")

        print("\n" + "-"*80 + "\n")

    # Get top 10 students by message count
    print("="*80)
    print("TOP 10 STUDENTS BY MESSAGE COUNT")
    print("="*80 + "\n")

    student_df = df[df['role'] == 'student']
    top_students = student_df.groupby('author_id').size().nlargest(10)

    for rank, (student_id, msg_count) in enumerate(top_students.items(), 1):
        student_msgs = student_df[student_df['author_id'] == student_id]

        # Get student name
        student_name = student_msgs.iloc[0]['author_name'] if 'author_name' in student_msgs.columns else f"Student_{student_id}"

        # Get conversation count
        conv_count = student_msgs['conversation_id'].nunique()

        # Sample 5 random messages
        sample_size = min(5, len(student_msgs))
        samples = student_msgs.sample(n=sample_size, random_state=42)

        print(f"#{rank}. {student_name} (ID: {student_id})")
        print(f"    {msg_count:,} messages across {conv_count:,} conversations")
        print(f"    Sample messages:\n")

        for i, (_, msg) in enumerate(samples.iterrows(), 1):
            text = msg['content'] if 'content' in msg else msg.get('text', '[no text field]')
            # Truncate long messages
            if len(text) > 150:
                text = text[:150] + "..."
            print(f"      [{i}] {text}")

        print("\n" + "-"*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
