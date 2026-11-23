#!/usr/bin/env python3
"""
Robust Error Classification for VotIE Predictions

Classifies prediction errors into four categories:
1. MIS (Missing) - Gold entities not predicted (False Negatives)
2. SPU (Spurious) - Predicted entities not in gold (False Positives)
3. INC_BOUNDARY - Correct type but wrong span boundaries
4. INC_TYPE - Correct span but wrong entity type

Usage:
    python scripts/error_classification.py --help
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.llm_extraction.shared.data_utils import load_jsonl


@dataclass
class Entity:
    """Entity with span and type information"""
    text: str
    type: str
    start: int
    end: int
    doc_id: str = ""

    def __hash__(self):
        return hash((self.start, self.end, self.type, self.doc_id))

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return (self.start == other.start and 
                self.end == other.end and 
                self.type == other.type and
                self.doc_id == other.doc_id)


@dataclass
class ErrorCase:
    """Represents a single error with full context"""
    category: str  # MIS, SPU, INC_BOUNDARY, INC_TYPE
    doc_id: str
    entity_type: str
    gold_entity: Entity = None
    pred_entity: Entity = None
    context: str = ""


class ErrorClassifier:
    """Classifies prediction errors into standard categories"""

    def __init__(self):
        self.entity_types = [
            'VOTER-FAVOR', 'VOTER-AGAINST', 'VOTER-ABSTENTION', 'VOTER-ABSENT',
            'VOTING', 'SUBJECT', 'COUNTING-UNANIMITY', 'COUNTING-MAJORITY'
        ]
        self.overlap_threshold = 0.5  # Minimum overlap to consider spans as matching

    def compute_span_overlap(self, ent1: Entity, ent2: Entity) -> float:
        """
        Compute character-level overlap ratio between two entities.
        Returns overlap as fraction of the smaller span.
        """
        overlap_start = max(ent1.start, ent2.start)
        overlap_end = min(ent1.end, ent2.end)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap_len = overlap_end - overlap_start
        min_len = min(ent1.end - ent1.start, ent2.end - ent2.start)
        
        return overlap_len / min_len if min_len > 0 else 0.0

    def has_exact_boundaries(self, ent1: Entity, ent2: Entity) -> bool:
        """Check if two entities have identical span boundaries"""
        return ent1.start == ent2.start and ent1.end == ent2.end

    def load_gold_standard_bio(self, gold_file: str) -> Dict[str, List[Entity]]:
        """Load gold standard entities from BIO format test file (for DeBERTa)"""
        gold_entities = defaultdict(list)
        data = load_jsonl(gold_file)
        
        for example in data:
            doc_id = example['id']
            tokens = example['tokens']
            labels = example['labels']  # Gold standard uses 'labels', not 'gold_labels'
            
            # Convert BIO labels to entities with character positions
            entities = self._extract_entities_from_bio(tokens, labels, doc_id)
            
            for entity in entities:
                entity.doc_id = doc_id
                gold_entities[doc_id].append(entity)
        
        return gold_entities

    def load_gold_standard_spans(self, gold_file: str) -> Dict[str, List[Entity]]:
        """Load gold standard entities from span format test file (for Gemini)"""
        gold_entities = defaultdict(list)
        data = load_jsonl(gold_file)
        
        for example in data:
            doc_id = example['id']
            
            for span in example.get('spans', []):
                entity = Entity(
                    text=span['text'],
                    type=span.get('label', span.get('type', 'UNKNOWN')),
                    start=span['start'],
                    end=span['end'],
                    doc_id=doc_id
                )
                gold_entities[doc_id].append(entity)
        
        return gold_entities

    def load_predictions(self, pred_file: str) -> Dict[str, List[Entity]]:
        """Load predicted entities from predictions file.

        Supports three formats:
        (a) ``tokens`` + ``pred_labels`` (legacy CRF/BiLSTM output)
        (b) ``subword_offsets`` + ``subword_pred_labels`` + ``text`` (current
            transformer pipeline; char-level offsets in the original text)
        (c) ``entities`` (LLM/Gemini output with text+type+start+end)
        """
        predictions = defaultdict(list)
        data = load_jsonl(pred_file)

        for example in data:
            doc_id = example['id']

            if 'subword_offsets' in example and 'subword_pred_labels' in example:
                entities = self._extract_entities_from_offsets(
                    text=example.get('text', ''),
                    offsets=example['subword_offsets'],
                    labels=example['subword_pred_labels'],
                    doc_id=doc_id,
                )
                predictions[doc_id].extend(entities)
            elif 'pred_labels' in example and 'tokens' in example:
                entities = self._extract_entities_from_bio(
                    tokens=example['tokens'],
                    labels=example['pred_labels'],
                    doc_id=doc_id,
                )
                predictions[doc_id].extend(entities)
            elif 'entities' in example:
                for ent in example.get('entities', []):
                    predictions[doc_id].append(Entity(
                        text=ent['text'],
                        type=ent['type'],
                        start=ent['start'],
                        end=ent['end'],
                        doc_id=doc_id,
                    ))

        return predictions

    def _extract_entities_from_offsets(
        self,
        text: str,
        offsets: List[List[int]],
        labels: List[str],
        doc_id: str,
    ) -> List[Entity]:
        """Convert subword BIO labels + char offsets into entity spans.

        Adjacent B-X / I-X subwords are merged. Span boundaries come from
        the first subword's start and the last subword's end in the original
        ``text`` (whitespace trimmed)."""
        entities: List[Entity] = []
        cur_type = None
        cur_start = None
        cur_end = None

        def flush():
            if cur_type is None:
                return
            s, e = cur_start, cur_end
            while s < e and s < len(text) and text[s].isspace():
                s += 1
            while e > s and e <= len(text) and text[e - 1].isspace():
                e -= 1
            entities.append(Entity(
                text=text[s:e] if text else '',
                type=cur_type,
                start=s,
                end=e,
                doc_id=doc_id,
            ))

        for (sw_start, sw_end), label in zip(offsets, labels):
            if label.startswith('B-'):
                flush()
                cur_type = label[2:]
                cur_start = sw_start
                cur_end = sw_end
            elif label.startswith('I-') and cur_type is not None and label[2:] == cur_type:
                cur_end = sw_end
            else:
                flush()
                cur_type = None
                cur_start = None
                cur_end = None
        flush()
        return entities
    
    def _extract_entities_from_bio(
        self,
        tokens: List[str],
        labels: List[str],
        doc_id: str
    ) -> List[Entity]:
        """Extract entity spans from BIO labels"""
        entities = []
        current_entity = []
        current_type = None
        current_start = None
        
        # Reconstruct character positions
        char_pos = 0
        token_positions = []
        
        for token in tokens:
            token_positions.append((char_pos, char_pos + len(token)))
            char_pos += len(token) + 1  # +1 for space
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith('B-'):
                # Save previous entity if exists
                if current_entity and current_type:
                    text = ' '.join(current_entity)
                    start_pos = token_positions[current_start][0]
                    end_pos = token_positions[i-1][1]
                    
                    entities.append(Entity(
                        text=text,
                        type=current_type,
                        start=start_pos,
                        end=end_pos,
                        doc_id=doc_id
                    ))
                
                # Start new entity
                current_entity = [token]
                current_type = label[2:]  # Remove 'B-'
                current_start = i
                
            elif label.startswith('I-') and current_entity:
                # Continue current entity
                current_entity.append(token)
                
            else:  # 'O' label
                # Save previous entity if exists
                if current_entity and current_type:
                    text = ' '.join(current_entity)
                    start_pos = token_positions[current_start][0]
                    end_pos = token_positions[i-1][1]
                    
                    entities.append(Entity(
                        text=text,
                        type=current_type,
                        start=start_pos,
                        end=end_pos,
                        doc_id=doc_id
                    ))
                
                current_entity = []
                current_type = None
                current_start = None
        
        # Handle final entity
        if current_entity and current_type:
            text = ' '.join(current_entity)
            start_pos = token_positions[current_start][0]
            end_pos = token_positions[len(tokens)-1][1]
            
            entities.append(Entity(
                text=text,
                type=current_type,
                start=start_pos,
                end=end_pos,
                doc_id=doc_id
            ))
        
        return entities

    def classify_errors(
        self,
        gold: Dict[str, List[Entity]],
        predictions: Dict[str, List[Entity]],
        texts: Dict[str, str]
    ) -> List[ErrorCase]:
        """
        Classify all errors into four categories:
        - MIS: Missing entities (gold but not predicted)
        - SPU: Spurious entities (predicted but not gold)
        - INC_BOUNDARY: Wrong boundaries (correct type, overlapping span)
        - INC_TYPE: Wrong type (same boundaries, wrong type)
        """
        errors = []

        # Iterate the union, not just gold. Gold is a defaultdict populated only
        # for documents that have spans, so looping over it alone skipped every
        # no-vote segment (68 of the 529 test documents) and silently dropped the
        # spurious predictions made inside them. Sorted for deterministic output.
        # NB: use gold.get(), not gold[], so a lookup cannot insert an empty key.
        all_doc_ids = sorted(set(gold) | set(predictions))

        for doc_id in all_doc_ids:
            gold_entities = gold.get(doc_id, [])
            pred_entities = predictions.get(doc_id, [])
            text = texts.get(doc_id, "")
            
            matched_gold = set()
            matched_pred = set()
            
            # First pass: Find matches and classify errors
            for g_idx, gold_ent in enumerate(gold_entities):
                best_pred_idx = None
                best_overlap = 0.0
                best_pred = None
                
                # Find best matching prediction
                for p_idx, pred_ent in enumerate(pred_entities):
                    if p_idx in matched_pred:
                        continue
                    
                    overlap = self.compute_span_overlap(gold_ent, pred_ent)
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_pred_idx = p_idx
                        best_pred = pred_ent
                
                # Classify based on best match
                if best_pred_idx is not None and best_overlap >= self.overlap_threshold:
                    matched_gold.add(g_idx)
                    matched_pred.add(best_pred_idx)
                    
                    # Check if exact match
                    if (self.has_exact_boundaries(gold_ent, best_pred) and 
                        gold_ent.type == best_pred.type):
                        # Perfect match - no error
                        continue
                    
                    # Check error type
                    if self.has_exact_boundaries(gold_ent, best_pred):
                        # Same boundaries, different type -> INC_TYPE
                        errors.append(ErrorCase(
                            category='INC_TYPE',
                            doc_id=doc_id,
                            entity_type=gold_ent.type,
                            gold_entity=gold_ent,
                            pred_entity=best_pred,
                            context=self._extract_context(text, gold_ent)
                        ))
                    elif gold_ent.type == best_pred.type:
                        # Same type, different boundaries -> INC_BOUNDARY
                        errors.append(ErrorCase(
                            category='INC_BOUNDARY',
                            doc_id=doc_id,
                            entity_type=gold_ent.type,
                            gold_entity=gold_ent,
                            pred_entity=best_pred,
                            context=self._extract_context(text, gold_ent)
                        ))
                    else:
                        # Different type AND boundaries - count as both errors
                        # This is a complex case - treat as boundary + type error
                        errors.append(ErrorCase(
                            category='INC_BOUNDARY',
                            doc_id=doc_id,
                            entity_type=gold_ent.type,
                            gold_entity=gold_ent,
                            pred_entity=best_pred,
                            context=self._extract_context(text, gold_ent)
                        ))
                        errors.append(ErrorCase(
                            category='INC_TYPE',
                            doc_id=doc_id,
                            entity_type=gold_ent.type,
                            gold_entity=gold_ent,
                            pred_entity=best_pred,
                            context=self._extract_context(text, gold_ent)
                        ))
                else:
                    # No match found -> MIS (Missing)
                    errors.append(ErrorCase(
                        category='MIS',
                        doc_id=doc_id,
                        entity_type=gold_ent.type,
                        gold_entity=gold_ent,
                        pred_entity=None,
                        context=self._extract_context(text, gold_ent)
                    ))
            
            # Second pass: Find spurious predictions
            for p_idx, pred_ent in enumerate(pred_entities):
                if p_idx not in matched_pred:
                    # Unmatched prediction -> SPU (Spurious)
                    errors.append(ErrorCase(
                        category='SPU',
                        doc_id=doc_id,
                        entity_type=pred_ent.type,
                        gold_entity=None,
                        pred_entity=pred_ent,
                        context=self._extract_context(text, pred_ent)
                    ))
        
        return errors

    def _extract_context(self, text: str, entity: Entity, context_chars: int = 50) -> str:
        """Extract surrounding context for an entity"""
        if not text:
            return ""
        
        start = max(0, entity.start - context_chars)
        end = min(len(text), entity.end + context_chars)
        
        context = text[start:end]
        
        # Highlight the entity
        entity_start_in_context = entity.start - start
        entity_end_in_context = entity.end - start
        
        if 0 <= entity_start_in_context < len(context):
            highlighted = (
                context[:entity_start_in_context] + 
                "**" + context[entity_start_in_context:entity_end_in_context] + "**" +
                context[entity_end_in_context:]
            )
            return highlighted
        
        return context

    def generate_statistics(
        self,
        errors: List[ErrorCase],
        model_name: str,
        max_examples: int = 5,
    ) -> Dict:
        """Generate comprehensive error statistics.

        max_examples caps retained examples per category; 0 keeps all of them,
        which is what the annotation sampler needs.
        """

        stats = {
            'model_name': model_name,
            'total_errors': len(errors),
            'error_breakdown': Counter(e.category for e in errors),
            'errors_by_type': defaultdict(lambda: Counter()),
            'examples_by_category': defaultdict(list)
        }

        # Count errors by entity type and category
        for error in errors:
            stats['errors_by_type'][error.entity_type][error.category] += 1

            if max_examples <= 0 or len(stats['examples_by_category'][error.category]) < max_examples:
                stats['examples_by_category'][error.category].append(error)

        return stats

    def generate_report(
        self,
        stats_list: List[Dict],
        output_file: str
    ):
        """Generate markdown report comparing models"""
        
        lines = []
        lines.append("# Error Classification Analysis\n\n")
        lines.append("Robust error classification into four categories:\n")
        lines.append("- **MIS** (Missing): Gold entities not predicted\n")
        lines.append("- **SPU** (Spurious): Predicted entities not in gold\n")
        lines.append("- **INC_BOUNDARY**: Correct type but wrong span boundaries\n")
        lines.append("- **INC_TYPE**: Correct span but wrong entity type\n\n")
        
        lines.append("---\n\n")
        
        # Overall comparison
        lines.append("## 1. Overall Error Distribution\n\n")
        lines.append("| Model | Total Errors | MIS | SPU | INC_BOUNDARY | INC_TYPE |\n")
        lines.append("|-------|--------------|-----|-----|--------------|----------|\n")
        
        for stats in stats_list:
            model = stats['model_name']
            total = stats['total_errors']
            breakdown = stats['error_breakdown']
            
            lines.append(f"| {model} | {total} | {breakdown['MIS']} | {breakdown['SPU']} | "
                        f"{breakdown['INC_BOUNDARY']} | {breakdown['INC_TYPE']} |\n")
        
        lines.append("\n")
        
        # Percentage breakdown
        lines.append("## 2. Error Distribution (Percentage)\n\n")
        lines.append("| Model | MIS % | SPU % | INC_BOUNDARY % | INC_TYPE % |\n")
        lines.append("|-------|-------|-------|----------------|------------|\n")
        
        for stats in stats_list:
            model = stats['model_name']
            total = stats['total_errors']
            breakdown = stats['error_breakdown']
            
            if total > 0:
                mis_pct = 100 * breakdown['MIS'] / total
                spu_pct = 100 * breakdown['SPU'] / total
                boundary_pct = 100 * breakdown['INC_BOUNDARY'] / total
                type_pct = 100 * breakdown['INC_TYPE'] / total
            else:
                mis_pct = spu_pct = boundary_pct = type_pct = 0.0
            
            lines.append(f"| {model} | {mis_pct:.1f}% | {spu_pct:.1f}% | "
                        f"{boundary_pct:.1f}% | {type_pct:.1f}% |\n")
        
        lines.append("\n")
        
        # Per-entity-type analysis
        lines.append("## 3. Errors by Entity Type\n\n")
        
        for stats in stats_list:
            model = stats['model_name']
            lines.append(f"### {model}\n\n")
            lines.append("| Entity Type | MIS | SPU | INC_BOUNDARY | INC_TYPE | Total |\n")
            lines.append("|-------------|-----|-----|--------------|----------|-------|\n")
            
            for etype in sorted(stats['errors_by_type'].keys()):
                counts = stats['errors_by_type'][etype]
                total_type = sum(counts.values())
                
                lines.append(f"| {etype} | {counts['MIS']} | {counts['SPU']} | "
                           f"{counts['INC_BOUNDARY']} | {counts['INC_TYPE']} | {total_type} |\n")
            
            lines.append("\n")
        
        # Example errors
        lines.append("## 4. Example Errors\n\n")
        
        for stats in stats_list:
            model = stats['model_name']
            lines.append(f"### {model}\n\n")
            
            for category in ['MIS', 'SPU', 'INC_BOUNDARY', 'INC_TYPE']:
                examples = stats['examples_by_category'][category]
                
                if not examples:
                    continue
                
                lines.append(f"#### {category}\n\n")
                
                for i, error in enumerate(examples[:5], 1):
                    lines.append(f"**Example {i}:**\n")
                    lines.append(f"- Document: `{error.doc_id}`\n")
                    lines.append(f"- Entity Type: `{error.entity_type}`\n")
                    
                    if error.gold_entity:
                        lines.append(f"- Gold: \"{error.gold_entity.text}\" "
                                   f"[{error.gold_entity.start}:{error.gold_entity.end}]\n")
                    else:
                        lines.append(f"- Gold: (none)\n")
                    
                    if error.pred_entity:
                        lines.append(f"- Pred: \"{error.pred_entity.text}\" "
                                   f"[{error.pred_entity.start}:{error.pred_entity.end}]\n")
                    else:
                        lines.append(f"- Pred: (none)\n")
                    
                    if error.context:
                        lines.append(f"- Context: ...{error.context}...\n")
                    
                    lines.append("\n")
                
                lines.append("\n")
        
        # Save report
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(''.join(lines))
        
        print(f"✓ Report saved to: {output_file}")

    def save_json_results(self, stats_list: List[Dict], output_file: str):
        """Save detailed results as JSON"""
        
        # Convert to serializable format
        results = {}
        for stats in stats_list:
            model = stats['model_name']
            
            # Convert examples to dicts
            examples_dict = {}
            for category, examples in stats['examples_by_category'].items():
                examples_dict[category] = [
                    {
                        'doc_id': e.doc_id,
                        'category': e.category,
                        'entity_type': e.entity_type,
                        'gold': {
                            'text': e.gold_entity.text,
                            'type': e.gold_entity.type,
                            'start': e.gold_entity.start,
                            'end': e.gold_entity.end
                        } if e.gold_entity else None,
                        'pred': {
                            'text': e.pred_entity.text,
                            'type': e.pred_entity.type,
                            'start': e.pred_entity.start,
                            'end': e.pred_entity.end
                        } if e.pred_entity else None,
                        'context': e.context
                    }
                    for e in examples
                ]
            
            results[model] = {
                'total_errors': stats['total_errors'],
                'error_breakdown': dict(stats['error_breakdown']),
                'errors_by_type': {
                    etype: dict(counts)
                    for etype, counts in stats['errors_by_type'].items()
                },
                'example_errors': examples_dict
            }
            # Multi-seed rows carry mean/std alongside the reference-seed counts
            # above, so Table 3's "±" is regenerated rather than asserted.
            for key in ('seeds', 'reference_seed', 'aggregate'):
                if key in stats:
                    results[model][key] = stats[key]
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ JSON results saved to: {output_file}")


CATEGORIES = ['MIS', 'SPU', 'INC_BOUNDARY', 'INC_TYPE']

# Default model set. Encoder rows expand to three seeds so the "±" in Table 3 is
# computed; the generative rows are single runs. Paths are globs resolved below.
DEFAULT_MODELS = [
    ('BERTimbau-CRF', 'results/standard/bert_crf/seed_s*_predictions.jsonl'),
    ('DeBERTa-CRF', 'results/standard/deberta_crf/seed_s*_predictions.jsonl'),
    ('XLM-R-CRF', 'results/standard/xlmr_crf/seed_s*_predictions.jsonl'),
    ('Gemini-Few-Shot', 'results/generative/gemini/few_shot.jsonl'),
    ('GPT-5.5-Few-Shot', 'results/generative/gpt5.5/few_shot.jsonl'),
    ('AMALIA-Few-Shot', 'results/generative/amalia/few_shot.jsonl'),
]

# Seed 42 is the reference run: per-type breakdowns and worked examples come from
# it, so they stay stable while the headline counts average over seeds.
REFERENCE_SEED = 's42'


def _seed_of(path: str) -> str:
    """Extract the seed tag from a prediction filename, or '' if unseeded."""
    m = re.search(r'seed_(s\d+)', Path(path).name)
    return m.group(1) if m else ''


def resolve_spec(spec: str) -> Tuple[str, List[str]]:
    """Parse a NAME=PATH[,PATH...] spec into (name, sorted file list).

    Each path may be a glob. Multiple paths are seeds of the same model.
    """
    if '=' not in spec:
        raise argparse.ArgumentTypeError(
            f"--predictions expects NAME=PATH[,PATH...], got: {spec!r}"
        )
    name, _, paths = spec.partition('=')
    files: List[str] = []
    for chunk in paths.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        matched = sorted(glob.glob(chunk))
        if not matched:
            if any(ch in chunk for ch in '*?['):
                raise FileNotFoundError(f"{name}: glob matched nothing: {chunk}")
            matched = [chunk]
        files.extend(matched)
    if not files:
        raise argparse.ArgumentTypeError(f"{name}: no prediction files given")
    # Reference seed first so it becomes the source of examples/per-type counts.
    files.sort(key=lambda p: (_seed_of(p) != REFERENCE_SEED, p))
    return name.strip(), files


def analyze_model(
    classifier: 'ErrorClassifier',
    name: str,
    pred_files: List[str],
    gold: Dict[str, List['Entity']],
    texts: Dict[str, str],
    max_examples: int,
    std_mode: str = 'sample',
    verbose: bool = True,
) -> Dict:
    """Classify errors for every seed of one model and aggregate the counts."""
    per_seed: List[Counter] = []
    reference_stats = None

    for path in pred_files:
        if not Path(path).exists():
            print(f"  ! missing, skipped: {path}")
            continue
        predictions = classifier.load_predictions(path)
        errors = classifier.classify_errors(gold, predictions, texts)
        counts = Counter(e.category for e in errors)
        counts['TOTAL'] = len(errors)
        per_seed.append(counts)

        if reference_stats is None:
            reference_stats = classifier.generate_statistics(errors, name, max_examples)

        if verbose:
            n_pred = sum(len(v) for v in predictions.values())
            print(f"  {Path(path).name:<40} {n_pred:>5} preds  {len(errors):>5} errors")

    if reference_stats is None:
        raise FileNotFoundError(f"{name}: none of the prediction files exist: {pred_files}")

    # Sample standard deviation (ddof=1): the seeds are a sample of the possible
    # random initialisations, not the population. Table 3 as published used it,
    # and every column untouched by the spurious-count fix reproduces exactly
    # under it and not under the population std. aggregate_seeds.py (Tables 1
    # and 2) now uses the same convention.
    spread = stdev if std_mode == 'sample' else pstdev

    aggregate = {}
    for key in ['TOTAL'] + CATEGORIES:
        values = [c[key] for c in per_seed]
        aggregate[key] = {
            'mean': mean(values),
            'std': spread(values) if len(values) > 1 else 0.0,
            'std_mode': std_mode,
            'n_seeds': len(values),
            'values': values,
        }

    reference_stats['seeds'] = [_seed_of(p) or 'single' for p in pred_files if Path(p).exists()]
    reference_stats['reference_seed'] = reference_stats['seeds'][0]
    reference_stats['aggregate'] = aggregate
    return reference_stats


def emit_latex(stats_list: List[Dict], output_file: str):
    """Emit the Table 3 body so the paper table is never hand-edited."""
    rows = []
    for stats in stats_list:
        agg = stats['aggregate']
        cells = []
        for key in ['TOTAL'] + CATEGORIES:
            m, s = agg[key]['mean'], agg[key]['std']
            if agg[key]['n_seeds'] > 1:
                cells.append(f"{m:.0f}$_{{\\pm {s:.0f}}}$")
            else:
                cells.append(f"{m:.0f}")
        rows.append(f"{stats['model_name']:<16} & " + " & ".join(cells) + r" \\")

    body = "\n".join(rows)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(
        "% Generated by scripts/error_classification.py -- do not edit by hand.\n"
        "% Columns: Model & Total & MIS & SPU & BND & TYP\n"
        f"{body}\n",
        encoding='utf-8',
    )
    print(f"✓ LaTeX table body saved to: {output_file}")


def main():
    """Run error classification analysis"""
    parser = argparse.ArgumentParser(
        description="Classify VotIE prediction errors into MIS / SPU / BND / TYP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # default model set, 3 seeds per encoder\n"
            "  python scripts/error_classification.py\n\n"
            "  # a single custom run, keeping every example for manual analysis\n"
            "  python scripts/error_classification.py --max-examples 0 \\\n"
            "      --predictions 'Gemini=results/generative/gemini/few_shot.jsonl'\n"
        ),
    )
    parser.add_argument(
        '--predictions', action='append', metavar='NAME=PATH[,PATH...]',
        help="Model to analyze. Repeatable. Paths may be globs; several paths "
             "are treated as seeds of one model. Defaults to the paper's model set.",
    )
    parser.add_argument('--gold', default='data/citilink-votie/test.jsonl',
                        help='Gold standard file (default: %(default)s)')
    parser.add_argument('--gold-format', choices=['spans', 'bio'], default='spans',
                        help='Gold annotation format (default: %(default)s)')
    parser.add_argument('--out-json', default='results/error_classification_results.json',
                        help='Output JSON path (default: %(default)s)')
    parser.add_argument('--out-md', default='results/error_classification_report.md',
                        help='Output markdown report path (default: %(default)s)')
    parser.add_argument('--out-latex', default=None,
                        help='Optional path for the Table 3 LaTeX body')
    parser.add_argument('--max-examples', type=int, default=5,
                        help='Examples kept per category; 0 keeps all (default: %(default)s)')
    parser.add_argument('--std', choices=['sample', 'population'], default='sample',
                        help="Spread reported across seeds. 'sample' (ddof=1) is what "
                             "the published Table 3 used; aggregate_seeds.py uses "
                             "'population' for Tables 1-2 (default: %(default)s)")

    args = parser.parse_args()

    specs = ([resolve_spec(s) for s in args.predictions]
             if args.predictions else
             [resolve_spec(f"{n}={p}") for n, p in DEFAULT_MODELS])

    print("=" * 80)
    print("VotIE Error Classification Analysis")
    print("=" * 80)

    classifier = ErrorClassifier()

    if args.gold_format == 'bio':
        gold = classifier.load_gold_standard_bio(args.gold)
    else:
        gold = classifier.load_gold_standard_spans(args.gold)

    gold_data = load_jsonl(args.gold)
    texts = {ex['id']: ex['text'] for ex in gold_data}
    total_gold = sum(len(entities) for entities in gold.values())
    n_empty = len(texts) - len(gold)
    print(f"Gold: {args.gold}")
    print(f"  {len(texts)} documents, {total_gold} entities, "
          f"{n_empty} with no gold span (still scored for spurious predictions)")

    stats_list = []
    for name, files in specs:
        print(f"\n{'='*80}\nAnalyzing: {name}\n{'='*80}")
        try:
            stats_list.append(
                analyze_model(classifier, name, files, gold, texts,
                              args.max_examples, args.std)
            )
        except FileNotFoundError as e:
            print(f"  ! {e}")

    if not stats_list:
        parser.error("no model produced results; check the --predictions paths")

    print(f"\n{'='*80}\nGenerating reports...\n{'='*80}")
    classifier.generate_report(stats_list, args.out_md)
    classifier.save_json_results(stats_list, args.out_json)
    if args.out_latex:
        emit_latex(stats_list, args.out_latex)

    print("\n## Summary Comparison (mean over seeds)")
    print(f"{'Model':<20} {'seeds':>5} {'Total':>12} {'MIS':>10} {'SPU':>10} "
          f"{'BND':>10} {'TYP':>9}")
    print("-" * 82)
    for stats in stats_list:
        agg = stats['aggregate']

        def cell(key, width):
            d = agg[key]
            txt = (f"{d['mean']:.0f}±{d['std']:.0f}" if d['n_seeds'] > 1
                   else f"{d['mean']:.0f}")
            return f"{txt:>{width}}"

        print(f"{stats['model_name']:<20} {agg['TOTAL']['n_seeds']:>5} "
              f"{cell('TOTAL', 12)} {cell('MIS', 10)} {cell('SPU', 10)} "
              f"{cell('INC_BOUNDARY', 10)} {cell('INC_TYPE', 9)}")


if __name__ == '__main__':
    main()
