"""
Hallucination Detection Module

Detects when LLMs generate unsupported claims by comparing outputs against
retrieved context using semantic similarity.

This module implements a three-tier grounding check:
1. Claim extraction (sentence splitting)
2. Semantic similarity scoring (sentence-transformers)
3. Confidence scoring per claim (0.0-1.0)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore


@dataclass
class Claim:
    """A single factual claim extracted from LLM output."""

    text: str  # The claim text
    sentence_index: int  # Position in original output (0-indexed)
    start_char: int  # Character offset in original text
    end_char: int  # End character offset


@dataclass
class ClaimGrounding:
    """Grounding assessment for a single claim."""

    claim: Claim
    is_grounded: bool  # True if supported by context
    confidence: float  # 0.0-1.0, confidence in grounding assessment
    similarity_score: float  # 0.0-1.0, max similarity to context chunks
    supporting_context: str | None  # Best matching context chunk
    context_index: int | None  # Index of supporting context in original list


@dataclass
class HallucinationAnalysis:
    """Complete hallucination analysis for an LLM response."""

    output_text: str  # Original LLM output
    context_chunks: list[str]  # Retrieved context used for comparison
    claims: list[ClaimGrounding]  # All extracted claims with grounding
    hallucination_detected: bool  # True if any ungrounded claims found
    overall_confidence: float  # 0.0-1.0, average confidence across claims
    ungrounded_claim_count: int  # Number of claims without support
    grounded_claim_count: int  # Number of claims with support


class HallucinationDetector:
    """Detects hallucinations in LLM outputs using grounding checks."""

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the hallucination detector.

        Args:
            similarity_threshold: Minimum similarity (0.0-1.0) to consider grounded
            model_name: Sentence-transformers model to use for embeddings
        """
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self._encoder = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._encoder = SentenceTransformer(model_name)
            except Exception:
                # Graceful degradation if model download fails
                self._encoder = None

    def analyze(
        self,
        output_text: str,
        context_chunks: list[str],
    ) -> HallucinationAnalysis:
        """
        Analyze LLM output for hallucinations against retrieved context.

        Args:
            output_text: The LLM's generated response
            context_chunks: List of retrieved context documents/chunks

        Returns:
            HallucinationAnalysis with per-claim grounding assessments
        """
        # Extract claims from output
        claims = self._extract_claims(output_text)

        # Assess grounding for each claim
        claim_groundings = []
        for claim in claims:
            grounding = self._assess_claim_grounding(claim, context_chunks)
            claim_groundings.append(grounding)

        # Calculate overall metrics
        ungrounded_claims = [c for c in claim_groundings if not c.is_grounded]
        grounded_claims = [c for c in claim_groundings if c.is_grounded]

        avg_confidence = (
            sum(c.confidence for c in claim_groundings) / len(claim_groundings)
            if claim_groundings
            else 1.0
        )

        return HallucinationAnalysis(
            output_text=output_text,
            context_chunks=context_chunks,
            claims=claim_groundings,
            hallucination_detected=len(ungrounded_claims) > 0,
            overall_confidence=avg_confidence,
            ungrounded_claim_count=len(ungrounded_claims),
            grounded_claim_count=len(grounded_claims),
        )

    def _extract_claims(self, text: str) -> list[Claim]:
        """
        Extract individual claims from text using sentence splitting.

        Args:
            text: Input text to extract claims from

        Returns:
            List of Claim objects with position information
        """
        if not text or not text.strip():
            return []

        # Split into sentences using regex
        # Handles periods, question marks, exclamation marks
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_pattern, text)

        claims = []
        char_offset = 0

        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # Find sentence in original text to get accurate offsets
            start_char = text.find(sentence, char_offset)
            end_char = start_char + len(sentence) if start_char != -1 else char_offset

            claims.append(
                Claim(
                    text=sentence,
                    sentence_index=idx,
                    start_char=start_char if start_char != -1 else char_offset,
                    end_char=end_char,
                )
            )

            char_offset = end_char

        return claims

    def _assess_claim_grounding(
        self,
        claim: Claim,
        context_chunks: list[str],
    ) -> ClaimGrounding:
        """
        Assess whether a claim is grounded in the provided context.

        Args:
            claim: The claim to assess
            context_chunks: Retrieved context to check against

        Returns:
            ClaimGrounding with similarity scores and support status
        """
        if not context_chunks:
            # No context provided - cannot verify grounding
            return ClaimGrounding(
                claim=claim,
                is_grounded=False,
                confidence=0.0,
                similarity_score=0.0,
                supporting_context=None,
                context_index=None,
            )

        # Use sentence-transformers if available, fallback to simple matching
        if self._encoder is not None:
            return self._assess_with_embeddings(claim, context_chunks)
        else:
            return self._assess_with_fallback(claim, context_chunks)

    def _assess_with_embeddings(
        self,
        claim: Claim,
        context_chunks: list[str],
    ) -> ClaimGrounding:
        """
        Assess grounding using sentence-transformers embeddings.

        Args:
            claim: The claim to assess
            context_chunks: Retrieved context to check against

        Returns:
            ClaimGrounding with semantic similarity scores
        """
        # Encode claim and context chunks
        claim_embedding = self._encoder.encode(claim.text, convert_to_tensor=True)
        context_embeddings = self._encoder.encode(
            context_chunks, convert_to_tensor=True
        )

        # Calculate cosine similarity
        import torch

        similarities = torch.nn.functional.cosine_similarity(
            claim_embedding.unsqueeze(0), context_embeddings
        )

        # Find best match
        max_similarity = float(similarities.max())
        best_match_idx = int(similarities.argmax())

        is_grounded = max_similarity >= self.similarity_threshold

        return ClaimGrounding(
            claim=claim,
            is_grounded=is_grounded,
            confidence=max_similarity,  # Use similarity as confidence
            similarity_score=max_similarity,
            supporting_context=context_chunks[best_match_idx]
            if is_grounded
            else None,
            context_index=best_match_idx if is_grounded else None,
        )

    def _assess_with_fallback(
        self,
        claim: Claim,
        context_chunks: list[str],
    ) -> ClaimGrounding:
        """
        Fallback grounding assessment using simple text matching.

        Used when sentence-transformers is not available.

        Args:
            claim: The claim to assess
            context_chunks: Retrieved context to check against

        Returns:
            ClaimGrounding with text-based similarity
        """
        import difflib

        claim_lower = claim.text.lower()
        best_similarity = 0.0
        best_match_idx = -1

        for idx, context in enumerate(context_chunks):
            context_lower = context.lower()

            # Use difflib for sequence matching
            similarity = difflib.SequenceMatcher(
                None, claim_lower, context_lower
            ).ratio()

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = idx

        is_grounded = best_similarity >= self.similarity_threshold

        return ClaimGrounding(
            claim=claim,
            is_grounded=is_grounded,
            confidence=best_similarity,
            similarity_score=best_similarity,
            supporting_context=context_chunks[best_match_idx]
            if is_grounded and best_match_idx >= 0
            else None,
            context_index=best_match_idx if is_grounded and best_match_idx >= 0 else None,
        )

    @staticmethod
    def extract_context_from_span(span_data: dict[str, Any]) -> list[str]:
        """
        Extract retrieved context chunks from a span.

        Looks for common attribute patterns:
        - retrieval.documents (list of dicts with 'text' or 'content')
        - retrieval.results (list of strings or dicts)
        - context (string or list)

        Args:
            span_data: Span dictionary with attributes

        Returns:
            List of context chunk strings
        """
        attributes = span_data.get("attributes", {})
        context_chunks = []

        # Try retrieval.documents (LlamaIndex, LangChain)
        documents = attributes.get("retrieval.documents", [])
        if isinstance(documents, list):
            for doc in documents:
                if isinstance(doc, dict):
                    text = doc.get("text") or doc.get("content") or doc.get("page_content")
                    if text:
                        context_chunks.append(str(text))
                elif isinstance(doc, str):
                    context_chunks.append(doc)

        # Try retrieval.results
        results = attributes.get("retrieval.results", [])
        if isinstance(results, list):
            for result in results:
                if isinstance(result, str):
                    context_chunks.append(result)
                elif isinstance(result, dict):
                    text = result.get("text") or result.get("content")
                    if text:
                        context_chunks.append(str(text))

        # Try generic context field
        context = attributes.get("context")
        if context:
            if isinstance(context, str):
                # Split by newlines if it looks like multiple chunks
                if "\n\n" in context:
                    context_chunks.extend(context.split("\n\n"))
                else:
                    context_chunks.append(context)
            elif isinstance(context, list):
                context_chunks.extend(str(c) for c in context if c)

        # Clean up and deduplicate
        context_chunks = [c.strip() for c in context_chunks if c and c.strip()]
        return list(dict.fromkeys(context_chunks))  # Deduplicate while preserving order
