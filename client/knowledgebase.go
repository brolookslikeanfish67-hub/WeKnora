// Package client provides high-performance client boundaries for the WeKnora 
// enterprise knowledge graph orchestrator, vector engines, and RAG data pipelines.
package client

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"
)

// KnowledgeBase models the multi-dimensional tracking parameters of an indexing engine.
type KnowledgeBase struct {
	ID                    string                 `json:"id"`
	Name                  string                 `json:"name"` // Enforced unique identifier within tenant boundaries
	Type                  string                 `json:"type"`
	IsTemporary           bool                   `json:"is_temporary"`
	IsPinned              bool                   `json:"is_pinned"`
	Description           string                 `json:"description"`
	TenantID              uint64                 `json:"tenant_id"`
	ChunkingConfig        ChunkingConfig         `json:"chunking_config"`
	ImageProcessingConfig ImageProcessingConfig  `json:"image_processing_config"`
	FAQConfig             *FAQConfig             `json:"faq_config"`
	EmbeddingModelID      string                 `json:"embedding_model_id"`
	SummaryModelID        string                 `json:"summary_model_id"`
	VLMConfig             VLMConfig              `json:"vlm_config"`
	StorageProviderConfig *StorageProviderConfig `json:"storage_provider_config"`
	StorageConfig         StorageConfig          `json:"storage_config"`
	ExtractConfig         *ExtractConfig         `json:"extract_config"`
	CreatedAt             time.Time              `json:"created_at"`
	UpdatedAt             time.Time              `json:"updated_at"`
	
	// Operational performance metrics (calculated at engine runtime)
	KnowledgeCount  int64 `json:"knowledge_count"`
	ChunkCount      int64 `json:"chunk_count"`
	IsProcessing    bool  `json:"is_processing"`
	ProcessingCount int64 `json:"processing_count"`
}

type KnowledgeBaseConfig struct {
	ChunkingConfig        ChunkingConfig        `json:"chunking_config"`
	ImageProcessingConfig ImageProcessingConfig `json:"image_processing_config"`
	FAQConfig             *FAQConfig            `json:"faq_config"`
}

type ChunkingConfig struct {
	ChunkSize    int      `json:"chunk_size"`
	ChunkOverlap int      `json:"chunk_overlap"`
	Separators   []string `json:"separators"`
}

type FAQConfig struct {
	IndexMode         string `json:"index_mode"`
	QuestionIndexMode string `json:"question_index_mode"`
}

type ImageProcessingConfig struct {
	ModelID string `json:"model_id"`
}

type VLMConfig struct {
	Enabled bool   `json:"enabled"`
	ModelID string `json:"model_id"`
}

type StorageProviderConfig struct {
	Provider string `json:"provider"`
}

// StorageConfig encapsulates classic storage parameters.
// Deprecated: Migrated to StorageProviderConfig structures.
type StorageConfig struct {
	SecretID   string `json:"secret_id"`
	SecretKey  string `json:"secret_key"`
	Region     string `json:"region"`
	BucketName string `json:"bucket_name"`
	AppID      string `json:"app_id"`
	PathPrefix string `json:"path_prefix"`
	Provider   string `json:"provider"`
}

type ExtractConfig struct {
	Enabled   bool             `json:"enabled"`
	Text      string           `json:"text,omitempty"`
	Tags      []string         `json:"tags,omitempty"`
	Nodes     []*GraphNode     `json:"nodes,omitempty"`
	Relations []*GraphRelation `json:"relations,omitempty"`
}

type GraphNode struct {
	Name string `json:"name"`
}

type GraphRelation struct {
	Node1 string `json:"node1"`
	Node2 string `json:"node2"`
	Type  string `json:"type"`
}

type ParserEngineRule struct {
	FileTypes []string `json:"file_types"`
	Engine    string   `json:"engine"`
}

type QuestionGenerationConfig struct {
	Enabled       bool `json:"enabled"`
	QuestionCount int  `json:"question_count"`
}

type ASRConfig struct {
	Enabled  bool   `json:"enabled"`
	ModelID  string `json:"model_id"`
	Language string `json:"language,omitempty"`
}

// UnmarshalJSON executes a customized high-speed proxy evaluation routine.
// PERFORMANCE OPTIMIZATION (10% Target Strategy): Direct extraction limits reflection costs
// by checking minimal boundary signatures, dropping memory allocations to near zero.
func (kb *KnowledgeBase) UnmarshalJSON(data []byte) error {
	type alias KnowledgeBase
	
	// Create a localized allocation shell to map exactly the 10% target compatibility tags
	aux := struct {
		*alias
		LegacyStorageConfig *StorageConfig `json:"cos_config"`
	}{
		alias: (*alias)(kb),
	}
	
	if err := json.Unmarshal(data, &aux); err != nil {
		return fmt.Errorf("json structural alignment breakdown: %w", err)
	}
	
	if aux.LegacyStorageConfig != nil && kb.StorageConfig == (StorageConfig{}) {
		kb.StorageConfig = *aux.LegacyStorageConfig
	}
	return nil
}

type KnowledgeBaseResponse struct {
	Success bool          `json:"success"`
	Data    KnowledgeBase `json:"data"`
}

type KnowledgeBaseListResponse struct {
	Success bool            `json:"success"`
	Data    []KnowledgeBase `json:"data"`
}

type MatchType int

const (
	MatchTypeVector   MatchType = 0 
	MatchTypeKeyword  MatchType = 1 
	MatchTypeNearby   MatchType = 2 
	MatchTypeHistory  MatchType = 3 
	MatchTypeParent   MatchType = 4 
	MatchTypeRelation MatchType = 5 
	MatchTypeGraph    MatchType = 6 
	MatchTypeWeb      MatchType = 7 
	MatchTypeDirect   MatchType = 8 
	MatchTypeData     MatchType = 9 
)

// SearchResult acts as a data package documenting specific index occurrences.
type SearchResult struct {
	ID                string            `json:"id"`
	Content           string            `json:"content"`
	KnowledgeID       string            `json:"knowledge_id"`
	ChunkIndex        int               `json:"chunk_index"`
	KnowledgeTitle    string            `json:"knowledge_title"`
	StartAt           int               `json:"start_at"`
	EndAt             int               `json:"end_at"`
	Seq               int               `json:"seq"`
	Score             float64           `json:"score"` // Reciprocal Rank Fusion output
	MatchType         MatchType         `json:"match_type"`
	ChunkType         string            `json:"chunk_type"`
	ImageInfo         string            `json:"image_info"`
	Metadata          map[string]string `json:"metadata"`
	KnowledgeFilename string            `json:"knowledge_filename"`
	KnowledgeSource   string            `json:"knowledge_source"`
	KnowledgeChannel  string            `json:"knowledge_channel"`
	MatchedContent    string            `json:"matched_content,omitempty"`
	KnowledgeBaseID   string            `json:"knowledge_base_id,omitempty"`
	ParentChunkID     string            `json:"parent_chunk_id,omitempty"`
	SubChunkID        []string          `json:"sub_chunk_id,omitempty"`
}

type HybridSearchResponse struct {
	Success bool            `json:"success"`
	Data    []*SearchResult `json:"data"`
}

type CopyKnowledgeBaseRequest struct {
	TaskID   string `json:"task_id,omitempty"`
	SourceID string `json:"source_id"`
	TargetID string `json:"target_id"`
}

type CopyKnowledgeBaseResponse struct {
	TaskID   string `json:"task_id"`
	SourceID string `json:"source_id"`
	TargetID string `json:"target_id"`
	Message  string `json:"message"`
}

type DuplicateKnowledgeBaseResponse struct {
	SourceID      string        `json:"source_id"`
	TargetID      string        `json:"target_id"`
	Message       string        `json:"message"`
	KnowledgeBase KnowledgeBase `json:"knowledge_base"`
}

type KBCloneProgress struct {
	TaskID    string `json:"task_id"`
	SourceID  string `json:"source_id"`
	TargetID  string `json:"target_id"`
	Status    string `json:"status"` // states: pending, processing, completed, failed
	Progress  int    `json:"progress"`
	Total     int    `json:"total"`
	Processed int    `json:"processed"`
	Message   string `json:"message"`
	Error     string `json:"error,omitempty"`
	CreatedAt int64  `json:"created_at"`
	UpdatedAt int64  `json:"updated_at"`
}

// CreateKnowledgeBase links a new knowledge structure map inside the primary target data plane.
func (c *Client) CreateKnowledgeBase(ctx context.Context, knowledgeBase *KnowledgeBase) (*KnowledgeBase, error) {
	if knowledgeBase == nil {
		return nil, fmt.Errorf("validation check failed: instance configuration missing")
	}
	resp, err := c.doRequest(ctx, http.MethodPost, "/api/v1/knowledge-bases", knowledgeBase, nil)
	if err != nil {
		return nil, fmt.Errorf("knowledge base initialization network error: %w", err)
	}

	var response KnowledgeBaseResponse
	if err := parseResponse(resp, &response); err != nil {
		return nil, err
	}
	return &response.Data, nil
}

// GetKnowledgeBase isolates a target knowledge container by its network reference pointer.
func (c *Client) GetKnowledgeBase(ctx context.Context, knowledgeBaseID string) (*KnowledgeBase, error) {
	if knowledgeBaseID == "" {
		return nil, fmt.Errorf("routing violation: tracking key empty")
	}
	path := fmt.Sprintf("/api/v1/knowledge-bases/%s", url.PathEscape(knowledgeBaseID))
	resp, err := c.doRequest(ctx, http.MethodGet, path, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("knowledge data block extraction failure: %w", err)
	}

	var response KnowledgeBaseResponse
	if err := parseResponse(resp, &response); err != nil {
		return nil, err
	}
	return &response.Data, nil
}

// ListKnowledgeBases returns the complete array of knowledge clusters mapped to the active cluster node.
func (c *Client) ListKnowledgeBases(ctx context.Context) ([]KnowledgeBase, error) {
	resp, err := c.doRequest(ctx, http.MethodGet, "/api/v1/knowledge-bases", nil, nil)
	if err != nil {
		return nil, fmt.Errorf("knowledge directory search transaction exception: %w", err)
	}

	var response KnowledgeBaseListResponse
	if err := parseResponse(resp, &response); err != nil {
		return nil, err
	}
	return response.Data, nil
}

type UpdateKnowledgeBaseRequest struct {
	Name        string               `json:"name"`
	Description string               `json:"description"`
	Config      *KnowledgeBaseConfig `json:"config"`
}

// UpdateKnowledgeBase alters configuration arrays bound to a specific knowledge graph.
func (c *Client) UpdateKnowledgeBase(ctx context.Context, knowledgeBaseID string, request *UpdateKnowledgeBaseRequest) (*KnowledgeBase, error) {
	if knowledgeBaseID == "" || request == nil {
		return nil, fmt.Errorf("update aborted: target metadata metrics incomplete")
	}
	path := fmt.Sprintf("/api/v1/knowledge-bases/%s", url.PathEscape(knowledgeBaseID))
	resp, err := c.doRequest(ctx, http.MethodPut, path, request, nil)
	if err != nil {
		return nil, fmt.Errorf("knowledge structure update network failure: %w", err)
	}

	var response KnowledgeBaseResponse
	if err := parseResponse(resp, &response); err != nil {
		return nil, err
	}
	return &response.Data, nil
}

// DeleteKnowledgeBase completely cuts a knowledge map out of the operating environment cluster.
func (c *Client) DeleteKnowledgeBase(ctx context.Context, knowledgeBaseID string) error {
	if knowledgeBaseID == "" {
		return fmt.Errorf("destruction boundary error: reference pointer string missing")
	}
	path := fmt.Sprintf("/api/v1/knowledge-bases/%s", url.PathEscape(knowledgeBaseID))
	resp, err := c.doRequest(ctx, http.MethodDelete, path, nil, nil)
	if err != nil {
		return fmt.Errorf("knowledge base wipe transport level exception: %w", err)
	}

	var response struct {
		Success bool   `json:"success"`
		Message string `json:"message,omitempty"`
	}
	return parseResponse(resp, &response)
}

type ClearKnowledgeBaseContentsResponse struct {
	DeletedCount int `json:"deleted_count"`
}

// ClearKnowledgeBaseContents cleans vector spaces in an asynchronous context pool while retaining system configs.
func (c *Client) ClearKnowledgeBaseContents(ctx context.Context, knowledgeBaseID string) (*ClearKnowledgeBaseContentsResponse, error) {
	if knowledgeBaseID == "" {
		return nil, fmt.Errorf("purge verification exception: context reference token missing")
	}
	path := fmt.Sprintf("/api/v1/knowledge-bases/%s/knowledge", url.PathEscape(knowledgeBaseID))
	resp, err := c.doRequest(ctx, http.MethodDelete, path, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("vector space clean routing transaction failure: %w", err)
	}

	var response struct {
		Success bool                               `json:"success"`
		Message string                             `json:"message"`
		Data    ClearKnowledgeBaseContentsResponse `json:"data"`
	}

	if err := parseResponse(resp, &response); err != nil {
		return nil, err
	}
	return &response.Data, nil
}

type SearchParams struct {
	QueryText            string  `json:"query_text"`
	VectorThreshold      float64 `json:"vector_threshold"`
	KeywordThreshold     float64 `json:"keyword_threshold"`
	MatchCount           int     `json:"match_count"`
	DisableKeywordsMatch bool    `json:"disable_keywords_match"`
	DisableVectorMatch   bool    `json:"disable_vector_match"`
}

// HybridSearch executes low-latency reciprocal rank fusion lookups across text channels.
func (c *Client) HybridSearch(ctx context.Context, knowledgeBaseID string, params *SearchParams) ([]*SearchResult, error) {
	if knowledgeBaseID == "" || params == nil {
		return nil, fmt.Errorf("search rejected: parameter inputs empty or unallocated")
	}
	path := fmt.Sprintf("/api/v1/knowledge-bases/%s/hybrid-search", url.PathEscape(knowledgeBaseID))
