use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use bip39::Language;

use crate::{
    domain::{
        Candidate, CandidateCursor, OrderMode, PermutationCursor, PhaseSummary, RecoveryRecipe,
        RecoverySettings, SearchPhase, SpacingMode, WrittenWords,
    },
    error::RecoverError,
};

/// A BIP39 neighbor and its edit distance from the written token
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeighborWord {
    /// Suggested BIP39 word
    pub word: String,
    /// Damerau-Levenshtein distance from the written token
    pub distance: usize,
}

/// Ranked BIP39 neighbors for one written token
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeighborSuggestion {
    /// Written token
    pub written: String,
    /// Nearest other BIP39 words
    pub neighbors: Vec<NeighborWord>,
}

/// Immutable, deterministic recovery search plan
#[derive(Debug, Clone)]
pub struct RecoveryPlan {
    settings: RecoverySettings,
    suggestions: Vec<NeighborSuggestion>,
    phases: Vec<PhasePlan>,
    token_trie: TokenTrie,
    base_lookup: HashMap<Vec<String>, BaseLocator>,
    canonical_check_required: bool,
}

#[derive(Debug, Clone)]
struct PhasePlan {
    phase: SearchPhase,
    bases: Vec<BasePlan>,
    count: u128,
}

#[derive(Debug, Clone)]
struct BasePlan {
    words: Vec<String>,
    multiset: Vec<String>,
    local_permutations: Vec<Vec<String>>,
    local_set: HashSet<Vec<String>>,
    local_ranks: HashMap<Vec<String>, usize>,
    lexical_count: u128,
}

#[derive(Debug, Clone)]
struct RankedBase {
    words: Vec<String>,
    total_distance: usize,
    positions: Vec<usize>,
    neighbor_ranks: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum CaseVariant {
    Lower,
    Title,
    Upper,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BaseLocator {
    replacements: usize,
    base_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum PermutationOrder {
    Local(usize),
    Lexical(u128),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct DerivationKey {
    phase_index: usize,
    base_index: usize,
    permutation: PermutationOrder,
    case_rank: u128,
    spacing_rank: u128,
}

struct CachedPermutation {
    phase: SearchPhase,
    base_index: usize,
    cursor: PermutationCursor,
    words: Vec<String>,
    order: PermutationOrder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TokenTerminal {
    word_index: usize,
    variants: u8,
}

#[derive(Debug, Clone, Default)]
struct TokenTrieNode {
    edges: BTreeMap<u8, usize>,
    terminals: Vec<TokenTerminal>,
}

#[derive(Debug, Clone, Default)]
struct TokenTrie {
    nodes: Vec<TokenTrieNode>,
    words: Vec<String>,
}

#[derive(Debug, Clone)]
struct ParsedSegmentation {
    words: Vec<String>,
    variants: Vec<u8>,
    spacing_mask: u128,
}

#[derive(Debug, Clone)]
struct LanguageSource {
    phase: SearchPhase,
    words: Vec<String>,
    counts: Vec<usize>,
    spacing: SpacingProfile,
    require_space: bool,
    written_order: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpacingProfile {
    Concatenated,
    Between,
    Coldcard,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ActiveToken {
    word_index: usize,
    variant: CaseVariant,
    offset: usize,
    leading_space: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct LanguageCursor {
    source: usize,
    remaining: Vec<usize>,
    active: Option<ActiveToken>,
    has_case: bool,
    emitted: usize,
    has_space: bool,
}

impl RecoveryPlan {
    /// Compile written words and settings into deterministic search phases
    pub fn compile(
        written_words: &WrittenWords,
        settings: RecoverySettings,
    ) -> Result<Self, RecoverError> {
        Self::compile_recipe(&RecoveryRecipe::from_written_words(written_words), settings)
    }

    /// Compile an advanced recipe and settings into deterministic search phases
    pub fn compile_recipe(
        recipe: &RecoveryRecipe,
        settings: RecoverySettings,
    ) -> Result<Self, RecoverError> {
        validate_settings(recipe, &settings)?;
        let primary_words = recipe
            .slots()
            .iter()
            .map(|slot| slot.alternatives()[0].clone())
            .collect::<Vec<_>>();
        let suggestions = nearest_words(&primary_words, settings.neighbors_per_word);
        let recipe_bases = recipe_bases(recipe)?;
        let mut phases = Vec::new();
        let mut seen_multisets = HashSet::new();
        let mut bases_by_replacement = Vec::new();
        for replacements in 0..=settings.max_replacements.min(recipe.slots().len()) {
            let mut bases = Vec::new();
            for written in &recipe_bases {
                if replacements > written.len() {
                    continue;
                }
                let base_suggestions = nearest_words(written, settings.neighbors_per_word);
                bases.extend(ranked_bases(
                    written,
                    &base_suggestions,
                    replacements,
                    &settings,
                    &mut seen_multisets,
                )?);
            }
            bases_by_replacement.push(bases);
        }

        for phase in SearchPhase::all(settings.max_replacements) {
            if settings.lowercase_already_tried && !phase.includes_case_variants() {
                continue;
            }
            if phase.replacement_count() > settings.max_replacements
                || phase.replacement_count() > recipe.slots().len()
            {
                continue;
            }
            let bases = bases_by_replacement[phase.replacement_count()].clone();
            phases.push(PhasePlan {
                phase,
                bases,
                count: 0,
            });
        }

        let unique_counts = unique_phase_counts(&phases, &settings)?;
        for phase in &mut phases {
            phase.count = unique_counts[phase.phase.index(settings.max_replacements)];
        }
        let (token_trie, base_lookup) = build_plan_indexes(&phases);
        let canonical_check_required = tokens_require_canonical_check(&token_trie.words)
            || recipe_bases.len() > 1
            || settings.spacing != SpacingMode::Concatenated;

        Ok(Self {
            settings,
            suggestions,
            phases,
            token_trie,
            base_lookup,
            canonical_check_required,
        })
    }

    /// Immutable recovery settings
    pub fn settings(&self) -> &RecoverySettings {
        &self.settings
    }

    /// Ranked word suggestions shown before a search begins
    pub fn neighbor_suggestions(&self) -> &[NeighborSuggestion] {
        &self.suggestions
    }

    /// Exact candidate counts for enabled phases
    pub fn phase_summaries(&self) -> Vec<PhaseSummary> {
        self.phases
            .iter()
            .map(|phase| PhaseSummary {
                phase: phase.phase,
                count: phase.count,
            })
            .collect()
    }

    /// Total candidates from the first phase through `through`
    pub fn count_through(&self, through: SearchPhase) -> Result<u128, RecoverError> {
        if through.replacement_count() > self.settings.max_replacements {
            return Err(RecoverError::DisabledPhase(through.to_string()));
        }

        self.phases
            .iter()
            .take_while(|phase| phase.phase <= through)
            .try_fold(0_u128, |total, phase| {
                total
                    .checked_add(phase.count)
                    .ok_or(RecoverError::CountOverflow)
            })
    }

    /// Generate up to `limit` candidates and advance the cursor to the next candidate
    pub fn next_batch(
        &self,
        cursor: &mut CandidateCursor,
        through: SearchPhase,
        limit: usize,
    ) -> Result<Vec<Candidate>, RecoverError> {
        if through.replacement_count() > self.settings.max_replacements {
            return Err(RecoverError::DisabledPhase(through.to_string()));
        }

        let mut candidates = Vec::with_capacity(limit);
        let mut cached_permutation = None;
        while candidates.len() < limit {
            let Some(candidate) = self.next_candidate(cursor, through, &mut cached_permutation)?
            else {
                break;
            };
            candidates.push(candidate);
        }
        Ok(candidates)
    }

    fn next_candidate(
        &self,
        cursor: &mut CandidateCursor,
        through: SearchPhase,
        cached_permutation: &mut Option<CachedPermutation>,
    ) -> Result<Option<Candidate>, RecoverError> {
        loop {
            if cursor.phase > through {
                return Ok(None);
            }
            let Some(phase) = self.phases.iter().find(|phase| phase.phase == cursor.phase) else {
                if !advance_phase(cursor, through, self.settings.max_replacements) {
                    return Ok(None);
                }
                continue;
            };
            if cursor.base_index >= phase.bases.len() {
                if !advance_phase(cursor, through, self.settings.max_replacements) {
                    return Ok(None);
                }
                continue;
            }

            let base = &phase.bases[cursor.base_index];
            let cache_matches = cached_permutation.as_ref().is_some_and(|cached| {
                cached.phase == phase.phase
                    && cached.base_index == cursor.base_index
                    && cached.cursor == cursor.permutation
            });
            if !cache_matches {
                let Some((words, order)) = current_permutation(base, &mut cursor.permutation)?
                else {
                    cursor.base_index += 1;
                    cursor.permutation = PermutationCursor::default();
                    cursor.case_rank = 0;
                    cursor.spacing_rank = 0;
                    continue;
                };
                *cached_permutation = Some(CachedPermutation {
                    phase: phase.phase,
                    base_index: cursor.base_index,
                    cursor: cursor.permutation.clone(),
                    words,
                    order,
                });
            }
            let cached = cached_permutation
                .as_ref()
                .expect("the current permutation was cached");

            let case_count =
                case_pattern_count(cached.words.len(), phase.phase.includes_case_variants())?;
            if cursor.case_rank >= case_count {
                cursor.case_rank = 0;
                cursor.spacing_rank = 0;
                advance_permutation(&mut cursor.permutation, base.local_permutations.len());
                continue;
            }

            let case_rank = cursor.case_rank;
            let words = if phase.phase.includes_case_variants() {
                apply_case_pattern(
                    &cached.words,
                    case_pattern_at(cached.words.len(), case_rank)?,
                )
            } else {
                cached.words.clone()
            };
            let spacing_patterns = spacing_patterns(
                words.len(),
                self.settings.spacing,
                self.settings.concatenated_already_tried,
            )?;
            if cursor.spacing_rank >= spacing_patterns.len() as u128 {
                cursor.case_rank += 1;
                cursor.spacing_rank = 0;
                continue;
            }
            let spacing_rank = cursor.spacing_rank;
            let spacing_mask = spacing_patterns[spacing_rank as usize];
            cursor.spacing_rank += 1;
            let passphrase = apply_spacing(&words, spacing_mask);
            if passphrase.len() > self.settings.max_passphrase_bytes {
                continue;
            }
            let current_key = DerivationKey {
                phase_index: phase.phase.index(self.settings.max_replacements),
                base_index: cursor.base_index,
                permutation: cached.order.clone(),
                case_rank,
                spacing_rank,
            };
            if self.canonical_check_required
                && self.canonical_derivation(&passphrase)? != current_key
            {
                continue;
            }
            cursor.completed = cursor
                .completed
                .checked_add(1)
                .ok_or(RecoverError::CountOverflow)?;
            return Ok(Some(Candidate::from_passphrase(
                phase.phase,
                passphrase,
                words,
            )));
        }
    }

    fn canonical_derivation(&self, passphrase: &str) -> Result<DerivationKey, RecoverError> {
        let segmentations = self.token_trie.parse(passphrase.as_bytes());
        let mut canonical = None;
        for segmentation in segmentations {
            let mut multiset = segmentation.words.clone();
            multiset.sort();
            let Some(locator) = self.base_lookup.get(&multiset) else {
                continue;
            };
            let base = self
                .phases
                .iter()
                .find(|phase| phase.phase.replacement_count() == locator.replacements)
                .and_then(|phase| phase.bases.get(locator.base_index))
                .ok_or(RecoverError::CountOverflow)?;
            let permutation = permutation_order(base, &segmentation.words)?;
            let Ok(spacing_rank) = spacing_rank(
                segmentation.words.len(),
                self.settings.spacing,
                self.settings.concatenated_already_tried,
                segmentation.spacing_mask,
            ) else {
                continue;
            };

            let all_lower = segmentation
                .variants
                .iter()
                .all(|variants| variants & variant_bit(CaseVariant::Lower) != 0);
            if let Some(phase) = self
                .phase_for(locator.replacements, false)
                .filter(|_| all_lower)
            {
                let key = DerivationKey {
                    phase_index: phase.index(self.settings.max_replacements),
                    base_index: locator.base_index,
                    permutation: permutation.clone(),
                    case_rank: 0,
                    spacing_rank,
                };
                canonical =
                    Some(canonical.map_or(key.clone(), |known: DerivationKey| known.min(key)));
            }

            if let (Some(case_rank), Some(phase)) = (
                minimum_case_rank(&segmentation.variants)?,
                self.phase_for(locator.replacements, true),
            ) {
                let key = DerivationKey {
                    phase_index: phase.index(self.settings.max_replacements),
                    base_index: locator.base_index,
                    permutation: permutation.clone(),
                    case_rank,
                    spacing_rank,
                };
                canonical = Some(canonical.map_or(key.clone(), |known| known.min(key)));
            }
        }
        canonical.ok_or(RecoverError::CountOverflow)
    }

    fn phase_for(&self, replacements: usize, variants: bool) -> Option<SearchPhase> {
        self.phases.iter().map(|phase| phase.phase).find(|phase| {
            phase.replacement_count() == replacements && phase.includes_case_variants() == variants
        })
    }
}

/// Expected random XFP hits for a candidate count
pub fn expected_xfp_collisions(count: u128) -> f64 {
    count as f64 / 4_294_967_296.0
}

/// Probability of at least one random four-byte XFP collision
pub fn xfp_collision_probability(count: u128) -> f64 {
    -(-expected_xfp_collisions(count)).exp_m1()
}

fn validate_settings(
    recipe: &RecoveryRecipe,
    settings: &RecoverySettings,
) -> Result<(), RecoverError> {
    if settings.neighbors_per_word == 0 || settings.neighbors_per_word > 2_047 {
        return Err(RecoverError::InvalidSetting(
            "neighbors-per-word must be between 1 and 2047".into(),
        ));
    }
    if settings.max_replacements > recipe.slots().len() {
        return Err(RecoverError::InvalidSetting(
            "max-replacements cannot exceed the number of recipe slots".into(),
        ));
    }
    if settings.max_passphrase_bytes == 0 {
        return Err(RecoverError::InvalidSetting(
            "max-passphrase-bytes must be positive".into(),
        ));
    }
    if settings.concatenated_already_tried
        && !matches!(settings.spacing, SpacingMode::Both | SpacingMode::Coldcard)
    {
        return Err(RecoverError::InvalidSetting(
            "concatenated-already-tried requires spacing=both or spacing=coldcard".into(),
        ));
    }
    if recipe.slots().len() > 100 {
        return Err(RecoverError::InvalidSetting(
            "a recovery recipe may contain at most 100 slots".into(),
        ));
    }
    Ok(())
}

fn recipe_bases(recipe: &RecoveryRecipe) -> Result<Vec<Vec<String>>, RecoverError> {
    fn choose(
        slots: &[crate::domain::TokenSlot],
        index: usize,
        current: &mut Vec<String>,
        alternatives: &mut Vec<usize>,
        omissions: usize,
        output: &mut Vec<(usize, Vec<usize>, Vec<String>)>,
    ) {
        if index == slots.len() {
            output.push((omissions, alternatives.clone(), current.clone()));
            return;
        }
        let slot = &slots[index];
        for (rank, value) in slot.alternatives().iter().enumerate() {
            current.push(value.clone());
            alternatives.push(rank);
            choose(slots, index + 1, current, alternatives, omissions, output);
            alternatives.pop();
            current.pop();
        }
        if slot.is_optional() {
            alternatives.push(usize::MAX);
            choose(
                slots,
                index + 1,
                current,
                alternatives,
                omissions + 1,
                output,
            );
            alternatives.pop();
        }
    }

    let mut ranked = Vec::new();
    choose(
        recipe.slots(),
        0,
        &mut Vec::new(),
        &mut Vec::new(),
        0,
        &mut ranked,
    );
    ranked.sort_by(|left, right| (&left.0, &left.1, &left.2).cmp(&(&right.0, &right.1, &right.2)));
    let mut seen = HashSet::new();
    Ok(ranked
        .into_iter()
        .map(|(_, _, words)| words)
        .filter(|words| seen.insert(words.clone()))
        .collect())
}

fn nearest_words(words: &[String], count: usize) -> Vec<NeighborSuggestion> {
    let bip39_words = Language::English.word_list();
    words
        .iter()
        .map(|written| {
            let mut neighbors = bip39_words
                .iter()
                .enumerate()
                .filter(|(_, candidate)| **candidate != written)
                .map(|(index, candidate)| {
                    (
                        strsim::damerau_levenshtein(written, candidate),
                        index,
                        *candidate,
                    )
                })
                .collect::<Vec<_>>();
            neighbors.sort_unstable_by_key(|(distance, index, _)| (*distance, *index));
            NeighborSuggestion {
                written: written.clone(),
                neighbors: neighbors
                    .into_iter()
                    .take(count)
                    .map(|(distance, _, word)| NeighborWord {
                        word: word.to_owned(),
                        distance,
                    })
                    .collect(),
            }
        })
        .collect()
}

fn ranked_bases(
    written: &[String],
    suggestions: &[NeighborSuggestion],
    replacements: usize,
    settings: &RecoverySettings,
    seen_multisets: &mut HashSet<Vec<String>>,
) -> Result<Vec<BasePlan>, RecoverError> {
    let mut ranked = Vec::new();
    choose_replacement_positions(
        written,
        suggestions,
        replacements,
        0,
        &mut Vec::new(),
        &mut ranked,
    );
    ranked.sort_by(|left, right| {
        (
            left.total_distance,
            &left.positions,
            &left.neighbor_ranks,
            &left.words,
        )
            .cmp(&(
                right.total_distance,
                &right.positions,
                &right.neighbor_ranks,
                &right.words,
            ))
    });

    let mut bases = Vec::new();
    for base in ranked {
        if base.words.iter().map(String::len).sum::<usize>() > settings.max_passphrase_bytes {
            continue;
        }
        let mut multiset = base.words.clone();
        multiset.sort();
        if !seen_multisets.insert(multiset.clone()) {
            continue;
        }
        let local_permutations = match settings.order {
            OrderMode::Written => vec![base.words.clone()],
            OrderMode::Permuted => local_permutations(&base.words, settings.local_swap_radius),
        };
        let local_set = local_permutations.iter().cloned().collect();
        let local_ranks = local_permutations
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, words)| (words, index))
            .collect();
        let lexical_count = match settings.order {
            OrderMode::Written => 0,
            OrderMode::Permuted => multiset_permutation_count(&base.words)?,
        };
        bases.push(BasePlan {
            words: base.words,
            multiset,
            local_permutations,
            local_set,
            local_ranks,
            lexical_count,
        });
    }
    Ok(bases)
}

fn choose_replacement_positions(
    written: &[String],
    suggestions: &[NeighborSuggestion],
    remaining: usize,
    next_position: usize,
    selected: &mut Vec<(usize, usize)>,
    output: &mut Vec<RankedBase>,
) {
    if remaining == 0 {
        let mut words = written.to_vec();
        let mut total_distance = 0;
        let mut positions = Vec::with_capacity(selected.len());
        let mut neighbor_ranks = Vec::with_capacity(selected.len());
        for &(position, neighbor_rank) in selected.iter() {
            let neighbor = &suggestions[position].neighbors[neighbor_rank];
            words[position] = neighbor.word.clone();
            total_distance += neighbor.distance;
            positions.push(position);
            neighbor_ranks.push(neighbor_rank);
        }
        output.push(RankedBase {
            words,
            total_distance,
            positions,
            neighbor_ranks,
        });
        return;
    }

    let final_start = written.len().saturating_sub(remaining);
    for position in next_position..=final_start {
        for neighbor_rank in 0..suggestions[position].neighbors.len() {
            selected.push((position, neighbor_rank));
            choose_replacement_positions(
                written,
                suggestions,
                remaining - 1,
                position + 1,
                selected,
                output,
            );
            selected.pop();
        }
    }
}

fn local_permutations(words: &[String], radius: usize) -> Vec<Vec<String>> {
    let start = words.to_vec();
    let mut seen = HashSet::from([start.clone()]);
    let mut output = vec![start.clone()];
    let mut queue = VecDeque::from([(start, 0_usize)]);

    while let Some((current, depth)) = queue.pop_front() {
        if depth == radius {
            continue;
        }
        for index in 0..current.len().saturating_sub(1) {
            let mut next = current.clone();
            next.swap(index, index + 1);
            if seen.insert(next.clone()) {
                output.push(next.clone());
                queue.push_back((next, depth + 1));
            }
        }
    }
    output
}

fn current_permutation(
    base: &BasePlan,
    cursor: &mut PermutationCursor,
) -> Result<Option<(Vec<String>, PermutationOrder)>, RecoverError> {
    loop {
        match cursor {
            PermutationCursor::Local { index } => {
                if let Some(words) = base.local_permutations.get(*index) {
                    return Ok(Some((words.clone(), PermutationOrder::Local(*index))));
                }
                *cursor = PermutationCursor::Lexical { rank: 0 };
            }
            PermutationCursor::Lexical { rank } => {
                if *rank >= base.lexical_count {
                    return Ok(None);
                }
                let words = unrank_multiset_permutation(&base.words, *rank)?;
                if base.local_set.contains(&words) {
                    *rank += 1;
                    continue;
                }
                return Ok(Some((words, PermutationOrder::Lexical(*rank))));
            }
        }
    }
}

fn advance_permutation(cursor: &mut PermutationCursor, local_count: usize) {
    match cursor {
        PermutationCursor::Local { index } if *index + 1 < local_count => *index += 1,
        PermutationCursor::Local { .. } => {
            *cursor = PermutationCursor::Lexical { rank: 0 };
        }
        PermutationCursor::Lexical { rank } => *rank += 1,
    }
}

fn advance_phase(
    cursor: &mut CandidateCursor,
    through: SearchPhase,
    max_replacements: usize,
) -> bool {
    let Some(next) = cursor.phase.next(max_replacements) else {
        return false;
    };
    if next > through {
        return false;
    }
    cursor.phase = next;
    cursor.base_index = 0;
    cursor.permutation = PermutationCursor::default();
    cursor.case_rank = 0;
    cursor.spacing_rank = 0;
    true
}

fn spacing_patterns(
    word_count: usize,
    mode: SpacingMode,
    concatenated_already_tried: bool,
) -> Result<Vec<u128>, RecoverError> {
    let combinations = 1_u128
        .checked_shl(
            word_count
                .try_into()
                .map_err(|_| RecoverError::CountOverflow)?,
        )
        .ok_or(RecoverError::CountOverflow)?;
    let between = if word_count <= 1 { 0 } else { combinations - 2 };
    let mut patterns = match mode {
        SpacingMode::Concatenated => vec![0],
        SpacingMode::Between => vec![between],
        SpacingMode::Both => vec![0, between],
        SpacingMode::Coldcard => {
            let mut patterns = (0..combinations).collect::<Vec<_>>();
            patterns.sort_by_key(|mask| {
                let preferred = if *mask == 0 {
                    0
                } else if *mask == between {
                    1
                } else {
                    2 + (*mask ^ between).count_ones()
                };
                (preferred, *mask)
            });
            patterns
        }
    };
    patterns.dedup();
    if concatenated_already_tried {
        patterns.retain(|mask| *mask != 0);
    }
    Ok(patterns)
}

fn spacing_rank(
    word_count: usize,
    mode: SpacingMode,
    concatenated_already_tried: bool,
    mask: u128,
) -> Result<u128, RecoverError> {
    spacing_patterns(word_count, mode, concatenated_already_tried)?
        .iter()
        .position(|pattern| *pattern == mask)
        .map(|rank| rank as u128)
        .ok_or(RecoverError::CountOverflow)
}

fn apply_spacing(words: &[String], mask: u128) -> String {
    let capacity = words.iter().map(String::len).sum::<usize>() + mask.count_ones() as usize;
    let mut output = String::with_capacity(capacity);
    for (index, word) in words.iter().enumerate() {
        if mask & (1_u128 << index) != 0 {
            output.push(' ');
        }
        output.push_str(word);
    }
    output
}

fn multiset_permutation_count(words: &[String]) -> Result<u128, RecoverError> {
    let mut frequencies = HashMap::<&str, usize>::new();
    for word in words {
        *frequencies.entry(word).or_default() += 1;
    }
    let numerator = factorial(words.len())?;
    let denominator = frequencies.values().try_fold(1_u128, |value, frequency| {
        value
            .checked_mul(factorial(*frequency)?)
            .ok_or(RecoverError::CountOverflow)
    })?;
    Ok(numerator / denominator)
}

fn unrank_multiset_permutation(
    words: &[String],
    mut rank: u128,
) -> Result<Vec<String>, RecoverError> {
    let mut frequencies = words.iter().cloned().fold(
        std::collections::BTreeMap::<String, usize>::new(),
        |mut frequencies, word| {
            *frequencies.entry(word).or_default() += 1;
            frequencies
        },
    );
    let mut output = Vec::with_capacity(words.len());

    for _ in 0..words.len() {
        let choices = frequencies.keys().cloned().collect::<Vec<_>>();
        let mut selected = None;
        for choice in choices {
            let frequency = frequencies.get_mut(&choice).expect("choice exists");
            *frequency -= 1;
            let remaining = frequencies.values().sum();
            let block = multiset_permutation_count_for_frequencies(
                remaining,
                frequencies.values().copied(),
            )?;
            if rank < block {
                selected = Some(choice);
                break;
            }
            rank -= block;
            *frequencies.get_mut(&choice).expect("choice exists") += 1;
        }
        let selected = selected.ok_or(RecoverError::CountOverflow)?;
        output.push(selected.clone());
        if frequencies[&selected] == 0 {
            frequencies.remove(&selected);
        }
    }
    Ok(output)
}

fn multiset_permutation_count_for_frequencies(
    item_count: usize,
    mut frequencies: impl Iterator<Item = usize>,
) -> Result<u128, RecoverError> {
    let denominator = frequencies.try_fold(1_u128, |value, frequency| {
        value
            .checked_mul(factorial(frequency)?)
            .ok_or(RecoverError::CountOverflow)
    })?;
    Ok(factorial(item_count)? / denominator)
}

fn factorial(value: usize) -> Result<u128, RecoverError> {
    (2..=value).try_fold(1_u128, |product, factor| {
        product
            .checked_mul(factor as u128)
            .ok_or(RecoverError::CountOverflow)
    })
}

fn case_pattern_count(word_count: usize, variants: bool) -> Result<u128, RecoverError> {
    if !variants {
        return Ok(1);
    }
    3_u128
        .checked_pow(
            word_count
                .try_into()
                .map_err(|_| RecoverError::CountOverflow)?,
        )
        .and_then(|count| count.checked_sub(1))
        .ok_or(RecoverError::CountOverflow)
}

fn case_pattern_at(word_count: usize, rank: u128) -> Result<Vec<CaseVariant>, RecoverError> {
    if rank == 0 {
        return Ok(vec![CaseVariant::Title; word_count]);
    }
    if rank == 1 {
        return Ok(vec![CaseVariant::Upper; word_count]);
    }

    let mut remaining_rank = rank - 2;
    for weight in 1..=word_count {
        let variants = 2_u128
            .checked_pow(weight.try_into().map_err(|_| RecoverError::CountOverflow)?)
            .ok_or(RecoverError::CountOverflow)?;
        let combinations = binomial(word_count, weight)?;
        let mut block = combinations
            .checked_mul(variants)
            .ok_or(RecoverError::CountOverflow)?;
        if weight == word_count {
            block = block.checked_sub(2).ok_or(RecoverError::CountOverflow)?;
        }
        if remaining_rank >= block {
            remaining_rank -= block;
            continue;
        }

        let (combination_rank, variant_rank) = if weight == word_count {
            (0, remaining_rank + 1)
        } else {
            (remaining_rank / variants, remaining_rank % variants)
        };
        let positions = unrank_combination(word_count, weight, combination_rank)?;
        let mut pattern = vec![CaseVariant::Lower; word_count];
        for (variant_index, position) in positions.into_iter().enumerate() {
            pattern[position] = if variant_rank & (1_u128 << variant_index) == 0 {
                CaseVariant::Title
            } else {
                CaseVariant::Upper
            };
        }
        return Ok(pattern);
    }
    Err(RecoverError::CountOverflow)
}

fn apply_case_pattern(words: &[String], pattern: Vec<CaseVariant>) -> Vec<String> {
    words
        .iter()
        .zip(pattern)
        .map(|(word, variant)| match variant {
            CaseVariant::Lower => word.clone(),
            CaseVariant::Title => {
                let mut bytes = word.as_bytes().to_vec();
                bytes[0].make_ascii_uppercase();
                String::from_utf8(bytes).expect("written words are ASCII")
            }
            CaseVariant::Upper => word.to_ascii_uppercase(),
        })
        .collect()
}

fn binomial(n: usize, k: usize) -> Result<u128, RecoverError> {
    let k = k.min(n - k);
    (0..k).try_fold(1_u128, |value, index| {
        value
            .checked_mul((n - index) as u128)
            .map(|product| product / (index + 1) as u128)
            .ok_or(RecoverError::CountOverflow)
    })
}

fn unrank_combination(n: usize, k: usize, mut rank: u128) -> Result<Vec<usize>, RecoverError> {
    let mut output = Vec::with_capacity(k);
    let mut start = 0;
    for selected in 0..k {
        for candidate in start..n {
            let remaining = k - selected - 1;
            let block = if remaining == 0 {
                1
            } else if n - candidate - 1 < remaining {
                0
            } else {
                binomial(n - candidate - 1, remaining)?
            };
            if rank < block {
                output.push(candidate);
                start = candidate + 1;
                break;
            }
            rank -= block;
        }
    }
    if output.len() != k {
        return Err(RecoverError::CountOverflow);
    }
    Ok(output)
}

fn tokens_require_canonical_check(tokens: &[String]) -> bool {
    tokens.iter().enumerate().any(|(index, token)| {
        token.len() <= 1
            || tokens
                .iter()
                .enumerate()
                .any(|(other_index, other)| index != other_index && other.starts_with(token))
    })
}

fn variant_bit(variant: CaseVariant) -> u8 {
    1 << match variant {
        CaseVariant::Lower => 0,
        CaseVariant::Title => 1,
        CaseVariant::Upper => 2,
    }
}

fn transformed_word(word: &str, variant: CaseVariant) -> Vec<u8> {
    match variant {
        CaseVariant::Lower => word.as_bytes().to_vec(),
        CaseVariant::Title => {
            let mut bytes = word.as_bytes().to_vec();
            bytes[0].make_ascii_uppercase();
            bytes
        }
        CaseVariant::Upper => word.to_ascii_uppercase().into_bytes(),
    }
}

fn build_plan_indexes(phases: &[PhasePlan]) -> (TokenTrie, HashMap<Vec<String>, BaseLocator>) {
    let mut words = BTreeSet::new();
    let mut base_lookup = HashMap::new();
    for phase in phases {
        for (base_index, base) in phase.bases.iter().enumerate() {
            words.extend(base.words.iter().cloned());
            base_lookup
                .entry(base.multiset.clone())
                .or_insert(BaseLocator {
                    replacements: phase.phase.replacement_count(),
                    base_index,
                });
        }
    }
    (TokenTrie::new(words.into_iter().collect()), base_lookup)
}

impl TokenTrie {
    fn new(words: Vec<String>) -> Self {
        let mut trie = Self {
            nodes: vec![TokenTrieNode::default()],
            words,
        };
        for word_index in 0..trie.words.len() {
            for variant in [CaseVariant::Lower, CaseVariant::Title, CaseVariant::Upper] {
                let bytes = transformed_word(&trie.words[word_index], variant);
                trie.insert(&bytes, word_index, variant);
            }
        }
        trie
    }

    fn insert(&mut self, bytes: &[u8], word_index: usize, variant: CaseVariant) {
        let mut node = 0;
        for &byte in bytes {
            let next = if let Some(next) = self.nodes[node].edges.get(&byte) {
                *next
            } else {
                let next = self.nodes.len();
                self.nodes.push(TokenTrieNode::default());
                self.nodes[node].edges.insert(byte, next);
                next
            };
            node = next;
        }
        if let Some(terminal) = self.nodes[node]
            .terminals
            .iter_mut()
            .find(|terminal| terminal.word_index == word_index)
        {
            terminal.variants |= variant_bit(variant);
        } else {
            self.nodes[node].terminals.push(TokenTerminal {
                word_index,
                variants: variant_bit(variant),
            });
        }
    }

    fn parse(&self, input: &[u8]) -> Vec<ParsedSegmentation> {
        let mut output = Vec::new();
        self.parse_from(input, 0, 0, &mut Vec::new(), &mut Vec::new(), &mut output);
        output
    }

    fn parse_from(
        &self,
        input: &[u8],
        offset: usize,
        spacing_mask: u128,
        words: &mut Vec<String>,
        variants: &mut Vec<u8>,
        output: &mut Vec<ParsedSegmentation>,
    ) {
        if offset == input.len() {
            output.push(ParsedSegmentation {
                words: words.clone(),
                variants: variants.clone(),
                spacing_mask,
            });
            return;
        }
        if words.len() >= 100 {
            return;
        }

        let (offset, spacing_mask) = if input[offset] == b' ' {
            (offset + 1, spacing_mask | (1_u128 << words.len()))
        } else {
            (offset, spacing_mask)
        };
        if offset >= input.len() {
            return;
        }

        let mut node = 0;
        for index in offset..input.len() {
            let Some(next) = self.nodes[node].edges.get(&input[index]) else {
                break;
            };
            node = *next;
            for terminal in &self.nodes[node].terminals {
                words.push(self.words[terminal.word_index].clone());
                variants.push(terminal.variants);
                self.parse_from(input, index + 1, spacing_mask, words, variants, output);
                variants.pop();
                words.pop();
            }
        }
    }
}

fn permutation_order(base: &BasePlan, words: &[String]) -> Result<PermutationOrder, RecoverError> {
    if let Some(index) = base.local_ranks.get(words) {
        return Ok(PermutationOrder::Local(*index));
    }
    Ok(PermutationOrder::Lexical(rank_multiset_permutation(words)?))
}

fn rank_multiset_permutation(words: &[String]) -> Result<u128, RecoverError> {
    let mut frequencies =
        words
            .iter()
            .cloned()
            .fold(BTreeMap::<String, usize>::new(), |mut frequencies, word| {
                *frequencies.entry(word).or_default() += 1;
                frequencies
            });
    let mut rank = 0_u128;
    for selected in words {
        let choices = frequencies.keys().cloned().collect::<Vec<_>>();
        for choice in choices {
            if choice >= *selected {
                break;
            }
            *frequencies.get_mut(&choice).expect("choice exists") -= 1;
            let remaining = frequencies
                .iter()
                .flat_map(|(word, count)| std::iter::repeat_n(word.clone(), *count))
                .collect::<Vec<_>>();
            rank = rank
                .checked_add(multiset_permutation_count(&remaining)?)
                .ok_or(RecoverError::CountOverflow)?;
            *frequencies.get_mut(&choice).expect("choice exists") += 1;
        }
        let frequency = frequencies
            .get_mut(selected)
            .ok_or(RecoverError::CountOverflow)?;
        *frequency -= 1;
        if *frequency == 0 {
            frequencies.remove(selected);
        }
    }
    Ok(rank)
}

fn minimum_case_rank(variants: &[u8]) -> Result<Option<u128>, RecoverError> {
    let title = variant_bit(CaseVariant::Title);
    let upper = variant_bit(CaseVariant::Upper);
    if variants.iter().all(|allowed| allowed & title != 0) {
        return Ok(Some(0));
    }
    if variants.iter().all(|allowed| allowed & upper != 0) {
        return Ok(Some(1));
    }

    let pattern = variants
        .iter()
        .map(|allowed| {
            if allowed & variant_bit(CaseVariant::Lower) != 0 {
                Ok(CaseVariant::Lower)
            } else if allowed & title != 0 {
                Ok(CaseVariant::Title)
            } else if allowed & upper != 0 {
                Ok(CaseVariant::Upper)
            } else {
                Err(RecoverError::CountOverflow)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if pattern.iter().all(|variant| *variant == CaseVariant::Lower) {
        return Ok(None);
    }
    Ok(Some(case_pattern_rank(&pattern)?))
}

fn case_pattern_rank(pattern: &[CaseVariant]) -> Result<u128, RecoverError> {
    if pattern.iter().all(|variant| *variant == CaseVariant::Title) {
        return Ok(0);
    }
    if pattern.iter().all(|variant| *variant == CaseVariant::Upper) {
        return Ok(1);
    }

    let positions = pattern
        .iter()
        .enumerate()
        .filter_map(|(index, variant)| (*variant != CaseVariant::Lower).then_some(index))
        .collect::<Vec<_>>();
    let weight = positions.len();
    let mut rank = 2_u128;
    for prior_weight in 1..weight {
        let variants = 2_u128
            .checked_pow(
                prior_weight
                    .try_into()
                    .map_err(|_| RecoverError::CountOverflow)?,
            )
            .ok_or(RecoverError::CountOverflow)?;
        rank = rank
            .checked_add(
                binomial(pattern.len(), prior_weight)?
                    .checked_mul(variants)
                    .ok_or(RecoverError::CountOverflow)?,
            )
            .ok_or(RecoverError::CountOverflow)?;
    }
    let variant_rank = positions
        .iter()
        .enumerate()
        .try_fold(0_u128, |rank, (bit, position)| {
            if pattern[*position] == CaseVariant::Upper {
                rank.checked_add(1_u128 << bit)
                    .ok_or(RecoverError::CountOverflow)
            } else {
                Ok(rank)
            }
        })?;
    if weight == pattern.len() {
        return rank
            .checked_add(
                variant_rank
                    .checked_sub(1)
                    .ok_or(RecoverError::CountOverflow)?,
            )
            .ok_or(RecoverError::CountOverflow);
    }
    let combinations_before = rank_combination(pattern.len(), &positions)?;
    let variants = 2_u128
        .checked_pow(weight.try_into().map_err(|_| RecoverError::CountOverflow)?)
        .ok_or(RecoverError::CountOverflow)?;
    rank.checked_add(
        combinations_before
            .checked_mul(variants)
            .and_then(|value| value.checked_add(variant_rank))
            .ok_or(RecoverError::CountOverflow)?,
    )
    .ok_or(RecoverError::CountOverflow)
}

fn rank_combination(n: usize, positions: &[usize]) -> Result<u128, RecoverError> {
    let mut rank = 0_u128;
    let mut start = 0;
    for (selected, &position) in positions.iter().enumerate() {
        for candidate in start..position {
            let remaining = positions.len() - selected - 1;
            let block = if remaining == 0 {
                1
            } else if n - candidate - 1 < remaining {
                0
            } else {
                binomial(n - candidate - 1, remaining)?
            };
            rank = rank.checked_add(block).ok_or(RecoverError::CountOverflow)?;
        }
        start = position + 1;
    }
    Ok(rank)
}

fn unique_phase_counts(
    phases: &[PhasePlan],
    settings: &RecoverySettings,
) -> Result<Vec<u128>, RecoverError> {
    let sources = language_sources(phases, settings);
    let mut state = sources
        .iter()
        .enumerate()
        .map(|(source, language)| LanguageCursor {
            source,
            remaining: language.counts.clone(),
            active: None,
            has_case: false,
            emitted: 0,
            has_space: false,
        })
        .collect::<Vec<_>>();
    state.sort();
    count_language_state(
        &state,
        &sources,
        settings.max_replacements,
        &mut HashMap::new(),
    )
}

fn language_sources(phases: &[PhasePlan], settings: &RecoverySettings) -> Vec<LanguageSource> {
    let profiles: &[SpacingProfile] = match settings.spacing {
        SpacingMode::Concatenated => &[SpacingProfile::Concatenated],
        SpacingMode::Between => &[SpacingProfile::Between],
        SpacingMode::Both => &[SpacingProfile::Concatenated, SpacingProfile::Between],
        SpacingMode::Coldcard => &[SpacingProfile::Coldcard],
    };
    phases
        .iter()
        .flat_map(|phase| {
            phase.bases.iter().flat_map(move |base| {
                let frequencies = base.words.iter().cloned().fold(
                    BTreeMap::<String, usize>::new(),
                    |mut frequencies, word| {
                        *frequencies.entry(word).or_default() += 1;
                        frequencies
                    },
                );
                let words = frequencies.keys().cloned().collect::<Vec<_>>();
                let written_order = (settings.order == OrderMode::Written).then(|| {
                    base.words
                        .iter()
                        .map(|word| {
                            words
                                .binary_search(word)
                                .expect("base words belong to their language")
                        })
                        .collect()
                });
                profiles.iter().map(move |spacing| LanguageSource {
                    phase: phase.phase,
                    words: words.clone(),
                    counts: frequencies.values().copied().collect(),
                    spacing: *spacing,
                    require_space: settings.concatenated_already_tried,
                    written_order: written_order.clone(),
                })
            })
        })
        .collect()
}

fn count_language_state(
    state: &[LanguageCursor],
    sources: &[LanguageSource],
    max_replacements: usize,
    memo: &mut HashMap<Vec<LanguageCursor>, Vec<u128>>,
) -> Result<Vec<u128>, RecoverError> {
    if let Some(counts) = memo.get(state) {
        return Ok(counts.clone());
    }
    let mut counts = vec![0_u128; 2 + max_replacements * 2];
    if let Some(phase) = state
        .iter()
        .filter(|cursor| language_cursor_accepts(cursor, &sources[cursor.source]))
        .map(|cursor| sources[cursor.source].phase)
        .min()
    {
        counts[phase.index(max_replacements)] = 1;
    }

    let mut symbols = BTreeSet::new();
    for cursor in state {
        symbols.extend(language_cursor_symbols(cursor, &sources[cursor.source]));
    }
    for symbol in symbols {
        let mut next = BTreeSet::new();
        for cursor in state {
            next.extend(advance_language_cursor(
                cursor,
                &sources[cursor.source],
                symbol,
            ));
        }
        if next.is_empty() {
            continue;
        }
        let child = count_language_state(
            &next.into_iter().collect::<Vec<_>>(),
            sources,
            max_replacements,
            memo,
        )?;
        for (count, child_count) in counts.iter_mut().zip(child) {
            *count = count
                .checked_add(child_count)
                .ok_or(RecoverError::CountOverflow)?;
        }
    }
    memo.insert(state.to_vec(), counts.clone());
    Ok(counts)
}

fn language_cursor_accepts(cursor: &LanguageCursor, source: &LanguageSource) -> bool {
    cursor.active.is_none()
        && cursor.remaining.iter().all(|count| *count == 0)
        && (!source.phase.includes_case_variants() || cursor.has_case)
        && (!source.require_space || cursor.has_space)
}

fn language_cursor_symbols(cursor: &LanguageCursor, source: &LanguageSource) -> BTreeSet<u8> {
    if let Some(active) = &cursor.active {
        if active.leading_space {
            return BTreeSet::from(*b" ");
        }
        return BTreeSet::from([transformed_word(
            &source.words[active.word_index],
            active.variant,
        )[active.offset]]);
    }
    let mut symbols = BTreeSet::new();
    for word_index in language_word_choices(cursor, source) {
        for variant in language_variants(source.phase) {
            for leading_space in spacing_choices(source.spacing, cursor.emitted) {
                if leading_space {
                    symbols.insert(b' ');
                } else {
                    symbols.insert(transformed_word(&source.words[word_index], variant)[0]);
                }
            }
        }
    }
    symbols
}

fn advance_language_cursor(
    cursor: &LanguageCursor,
    source: &LanguageSource,
    symbol: u8,
) -> Vec<LanguageCursor> {
    if let Some(active) = &cursor.active {
        if active.leading_space {
            if symbol != b' ' {
                return Vec::new();
            }
            let mut next = cursor.clone();
            next.has_space = true;
            if let Some(next_active) = &mut next.active {
                next_active.leading_space = false;
                next_active.offset = 0;
            }
            return vec![next];
        }
        let bytes = transformed_word(&source.words[active.word_index], active.variant);
        if bytes[active.offset] != symbol {
            return Vec::new();
        }
        let mut next = cursor.clone();
        if active.offset + 1 == bytes.len() {
            next.active = None;
        } else if let Some(next_active) = &mut next.active {
            next_active.offset += 1;
        }
        return vec![next];
    }

    let mut output = Vec::new();
    for word_index in language_word_choices(cursor, source) {
        for variant in language_variants(source.phase) {
            let bytes = transformed_word(&source.words[word_index], variant);
            for leading_space in spacing_choices(source.spacing, cursor.emitted) {
                if (leading_space && symbol != b' ') || (!leading_space && bytes[0] != symbol) {
                    continue;
                }
                let mut next = cursor.clone();
                next.remaining[word_index] -= 1;
                next.has_case |= variant != CaseVariant::Lower;
                next.emitted += 1;
                if leading_space || bytes.len() > 1 {
                    next.active = Some(ActiveToken {
                        word_index,
                        variant,
                        offset: usize::from(!leading_space),
                        leading_space,
                    });
                }
                output.push(next);
            }
        }
    }
    output
}

fn language_word_choices(cursor: &LanguageCursor, source: &LanguageSource) -> Vec<usize> {
    if let Some(order) = &source.written_order {
        return order.get(cursor.emitted).copied().into_iter().collect();
    }
    cursor
        .remaining
        .iter()
        .enumerate()
        .filter_map(|(index, count)| (*count > 0).then_some(index))
        .collect()
}

fn spacing_choices(profile: SpacingProfile, emitted: usize) -> impl Iterator<Item = bool> {
    let choices = match profile {
        SpacingProfile::Concatenated => [Some(false), None],
        SpacingProfile::Between => [Some(emitted > 0), None],
        SpacingProfile::Coldcard => [Some(false), Some(true)],
    };
    choices.into_iter().flatten()
}

fn language_variants(phase: SearchPhase) -> impl Iterator<Item = CaseVariant> {
    let variants = if phase.includes_case_variants() {
        [
            Some(CaseVariant::Lower),
            Some(CaseVariant::Title),
            Some(CaseVariant::Upper),
        ]
    } else {
        [Some(CaseVariant::Lower), None, None]
    };
    variants.into_iter().flatten()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn written_phases_have_exact_disjoint_counts() {
        let words = WrittenWords::new(vec!["alpha".into(), "brisk".into()]).unwrap();
        let plan = RecoveryPlan::compile(&words, RecoverySettings::default()).unwrap();
        let summaries = plan.phase_summaries();

        assert_eq!(summaries[0].count, 2);
        assert_eq!(summaries[1].count, 16);

        let mut cursor = CandidateCursor::default();
        let candidates = plan
            .next_batch(&mut cursor, SearchPhase::WrittenCase, 100)
            .unwrap();
        let passphrases = candidates
            .iter()
            .map(|candidate| candidate.passphrase().to_owned())
            .collect::<HashSet<_>>();
        assert_eq!(candidates.len(), 18);
        assert_eq!(passphrases.len(), 18);
        assert!(!candidates
            .iter()
            .any(|candidate| candidate.passphrase().contains(' ')));
    }

    #[test]
    fn prefix_free_fast_path_matches_canonical_enumeration() {
        let words = WrittenWords::new(vec![
            "alpha".into(),
            "brisk".into(),
            "cactus".into(),
            "daring".into(),
        ])
        .unwrap();
        let fast = RecoveryPlan::compile(&words, RecoverySettings::default()).unwrap();
        assert!(!fast.canonical_check_required);
        let mut reference = fast.clone();
        reference.canonical_check_required = true;
        let mut fast_cursor = CandidateCursor::default();
        let mut reference_cursor = CandidateCursor::default();

        let fast_candidates = fast
            .next_batch(&mut fast_cursor, SearchPhase::WrittenCase, 4_096)
            .unwrap();
        let reference_candidates = reference
            .next_batch(&mut reference_cursor, SearchPhase::WrittenCase, 4_096)
            .unwrap();

        assert_eq!(fast_cursor, reference_cursor);
        assert_eq!(
            fast_candidates
                .iter()
                .map(Candidate::passphrase)
                .collect::<Vec<_>>(),
            reference_candidates
                .iter()
                .map(Candidate::passphrase)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn local_permutations_start_with_nearby_moves() {
        let words = vec!["a".into(), "b".into(), "c".into()];
        let permutations = local_permutations(&words, 1);

        assert_eq!(permutations[0], vec!["a", "b", "c"]);
        assert_eq!(permutations[1], vec!["b", "a", "c"]);
        assert_eq!(permutations[2], vec!["a", "c", "b"]);
    }

    #[test]
    fn nearest_words_use_bip39_order_to_break_distance_ties() {
        let words = WrittenWords::new(vec!["wive".into()]).unwrap();
        let suggestions = nearest_words(words.as_slice(), 3);

        assert_eq!(
            suggestions[0]
                .neighbors
                .iter()
                .map(|neighbor| (&*neighbor.word, neighbor.distance))
                .collect::<Vec<_>>(),
            [("give", 1), ("live", 1), ("wave", 1)]
        );
    }

    #[test]
    fn case_ranking_prefers_uniform_title_and_upper() {
        assert_eq!(
            case_pattern_at(2, 0).unwrap(),
            vec![CaseVariant::Title, CaseVariant::Title]
        );
        assert_eq!(
            case_pattern_at(2, 1).unwrap(),
            vec![CaseVariant::Upper, CaseVariant::Upper]
        );
        let patterns = (0..8)
            .map(|rank| case_pattern_at(2, rank).unwrap())
            .collect::<HashSet<_>>();
        assert_eq!(patterns.len(), 8);
    }

    #[test]
    fn multiset_unranking_is_unique_and_lexical() {
        let words = vec!["b".into(), "a".into(), "a".into()];
        let permutations = (0..3)
            .map(|rank| unrank_multiset_permutation(&words, rank).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            permutations,
            vec![
                vec!["a", "a", "b"],
                vec!["a", "b", "a"],
                vec!["b", "a", "a"]
            ]
        );
    }

    #[test]
    fn serialized_cursor_resumes_at_the_exact_next_candidate() {
        let words = WrittenWords::new(vec!["alpha".into(), "brisk".into()]).unwrap();
        let plan = RecoveryPlan::compile(&words, RecoverySettings::default()).unwrap();
        let mut uninterrupted_cursor = CandidateCursor::default();
        let uninterrupted = plan
            .next_batch(&mut uninterrupted_cursor, SearchPhase::NeighborLower(1), 32)
            .unwrap();

        let mut interrupted_cursor = CandidateCursor::default();
        let prefix = plan
            .next_batch(&mut interrupted_cursor, SearchPhase::NeighborLower(1), 7)
            .unwrap();
        let checkpoint = serde_json::to_vec(&interrupted_cursor).unwrap();
        let mut resumed_cursor: CandidateCursor = serde_json::from_slice(&checkpoint).unwrap();
        let suffix = plan
            .next_batch(&mut resumed_cursor, SearchPhase::NeighborLower(1), 25)
            .unwrap();

        let resumed_ids = prefix
            .iter()
            .chain(&suffix)
            .map(|candidate| candidate.id().clone())
            .collect::<Vec<_>>();
        let uninterrupted_ids = uninterrupted
            .iter()
            .map(|candidate| candidate.id().clone())
            .collect::<Vec<_>>();
        assert_eq!(resumed_ids, uninterrupted_ids);
        assert_eq!(resumed_cursor, uninterrupted_cursor);
    }

    #[test]
    fn replacement_and_case_phases_do_not_repeat_passphrase_bytes() {
        let words = WrittenWords::new(vec!["alpha".into(), "brisk".into()]).unwrap();
        let plan = RecoveryPlan::compile(&words, RecoverySettings::default()).unwrap();
        let expected = plan.count_through(SearchPhase::NeighborCase(2)).unwrap();
        let mut cursor = CandidateCursor::default();
        let candidates = plan
            .next_batch(&mut cursor, SearchPhase::NeighborCase(2), expected as usize)
            .unwrap();
        let unique = candidates
            .iter()
            .map(Candidate::passphrase)
            .collect::<HashSet<_>>();

        assert_eq!(candidates.len() as u128, expected);
        assert_eq!(unique.len(), candidates.len());
    }

    #[test]
    fn ambiguous_word_boundaries_emit_only_the_earliest_candidate() {
        let words = WrittenWords::new(vec!["car".into(), "dice".into()]).unwrap();
        let plan = RecoveryPlan::compile(&words, RecoverySettings::default()).unwrap();
        let expected = plan.count_through(SearchPhase::NeighborCase(2)).unwrap();
        let mut cursor = CandidateCursor::default();
        let candidates = plan
            .next_batch(&mut cursor, SearchPhase::NeighborCase(2), expected as usize)
            .unwrap();
        let collisions = candidates
            .iter()
            .filter(|candidate| candidate.passphrase() == "cardice")
            .collect::<Vec<_>>();

        assert_eq!(candidates.len() as u128, expected);
        assert_eq!(collisions.len(), 1);
        assert_eq!(collisions[0].phase(), SearchPhase::WrittenLower);
        assert_eq!(collisions[0].words(), ["car", "dice"]);
    }

    #[test]
    fn identical_title_and_upper_bytes_are_counted_once() {
        let words = WrittenWords::new(vec!["a".into()]).unwrap();
        let settings = RecoverySettings {
            max_replacements: 0,
            ..RecoverySettings::default()
        };
        let plan = RecoveryPlan::compile(&words, settings).unwrap();
        let summaries = plan.phase_summaries();
        assert_eq!(summaries[0].count, 1);
        assert_eq!(summaries[1].count, 1);

        let mut cursor = CandidateCursor::default();
        let candidates = plan
            .next_batch(&mut cursor, SearchPhase::WrittenCase, 10)
            .unwrap();
        assert_eq!(
            candidates
                .iter()
                .map(Candidate::passphrase)
                .collect::<Vec<_>>(),
            ["a", "A"]
        );
    }

    #[test]
    fn candidates_over_the_coldcard_limit_are_excluded_from_counts() {
        let words = WrittenWords::new(vec!["alphabet".into(), "another".into()]).unwrap();
        let settings = RecoverySettings {
            max_replacements: 0,
            max_passphrase_bytes: 10,
            ..RecoverySettings::default()
        };
        let plan = RecoveryPlan::compile(&words, settings).unwrap();

        assert_eq!(plan.count_through(SearchPhase::WrittenCase).unwrap(), 0);
    }

    #[test]
    fn completed_lowercase_work_is_excluded_from_the_plan() {
        let words = WrittenWords::new(vec!["alpha".into(), "brisk".into()]).unwrap();
        let settings = RecoverySettings {
            lowercase_already_tried: true,
            ..RecoverySettings::default()
        };
        let plan = RecoveryPlan::compile(&words, settings).unwrap();
        let phases = plan
            .phase_summaries()
            .into_iter()
            .map(|summary| summary.phase)
            .collect::<Vec<_>>();

        assert_eq!(
            phases,
            [
                SearchPhase::WrittenCase,
                SearchPhase::NeighborCase(1),
                SearchPhase::NeighborCase(2),
            ]
        );
        let mut cursor = CandidateCursor::default();
        let first = plan
            .next_batch(&mut cursor, SearchPhase::WrittenCase, 1)
            .unwrap();
        assert_eq!(first[0].passphrase(), "AlphaBrisk");
    }

    #[test]
    fn coldcard_spacing_covers_every_leading_space_pattern() {
        let words = WrittenWords::new(vec!["alpha".into(), "brisk".into()]).unwrap();
        let settings = RecoverySettings {
            max_replacements: 0,
            spacing: SpacingMode::Coldcard,
            ..RecoverySettings::default()
        };
        let plan = RecoveryPlan::compile(&words, settings).unwrap();
        assert_eq!(plan.phase_summaries()[0].count, 8);

        let mut cursor = CandidateCursor::default();
        let candidates = plan
            .next_batch(&mut cursor, SearchPhase::WrittenLower, 8)
            .unwrap();
        let passphrases = candidates
            .iter()
            .map(Candidate::passphrase)
            .collect::<HashSet<_>>();
        assert_eq!(passphrases.len(), 8);
        assert!(passphrases.contains("alphabrisk"));
        assert!(passphrases.contains("alpha brisk"));
        assert!(passphrases.contains(" alphabrisk"));
        assert!(passphrases.contains(" alpha brisk"));
    }

    #[test]
    fn completed_concatenated_pattern_is_excluded_only_from_spacing() {
        let words = WrittenWords::new(vec!["alpha".into(), "brisk".into()]).unwrap();
        let settings = RecoverySettings {
            max_replacements: 0,
            spacing: SpacingMode::Coldcard,
            concatenated_already_tried: true,
            ..RecoverySettings::default()
        };
        let plan = RecoveryPlan::compile(&words, settings).unwrap();
        assert_eq!(plan.phase_summaries()[0].count, 6);

        let mut cursor = CandidateCursor::default();
        let candidates = plan
            .next_batch(&mut cursor, SearchPhase::WrittenLower, 10)
            .unwrap();
        assert!(candidates
            .iter()
            .all(|candidate| candidate.passphrase().contains(' ')));
    }

    #[test]
    fn recipe_alternatives_and_optional_slots_are_ranked_and_deduplicated() {
        use crate::domain::TokenSlot;

        let recipe = RecoveryRecipe::new(vec![
            TokenSlot::new(vec!["alpha".into(), "alps".into()], false).unwrap(),
            TokenSlot::new(vec!["brisk".into()], true).unwrap(),
        ])
        .unwrap();
        let settings = RecoverySettings {
            max_replacements: 0,
            order: OrderMode::Written,
            ..RecoverySettings::default()
        };
        let plan = RecoveryPlan::compile_recipe(&recipe, settings).unwrap();
        assert_eq!(plan.phase_summaries()[0].count, 4);

        let mut cursor = CandidateCursor::default();
        let candidates = plan
            .next_batch(&mut cursor, SearchPhase::WrittenLower, 10)
            .unwrap();
        assert_eq!(
            candidates
                .iter()
                .map(Candidate::passphrase)
                .collect::<Vec<_>>(),
            ["alphabrisk", "alpsbrisk", "alpha", "alps"]
        );
    }

    #[test]
    fn replacement_phases_extend_to_the_configured_slot_count() {
        let words =
            WrittenWords::new(vec!["alpha".into(), "brisk".into(), "cactus".into()]).unwrap();
        let settings = RecoverySettings {
            neighbors_per_word: 1,
            max_replacements: 3,
            order: OrderMode::Written,
            ..RecoverySettings::default()
        };
        let plan = RecoveryPlan::compile(&words, settings).unwrap();
        let summaries = plan.phase_summaries();
        assert!(
            summaries.iter().any(|summary| {
                summary.phase == SearchPhase::NeighborLower(3) && summary.count > 0
            }),
            "{summaries:?}"
        );

        let limit = plan.count_through(SearchPhase::NeighborLower(3)).unwrap() as usize;
        let mut cursor = CandidateCursor::default();
        let candidates = plan
            .next_batch(&mut cursor, SearchPhase::NeighborLower(3), limit)
            .unwrap();
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.phase() == SearchPhase::NeighborLower(3)),
            "summaries={summaries:?} limit={limit}"
        );
    }
}
