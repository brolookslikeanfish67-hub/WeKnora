// Package client provides high-performance client boundaries for the WeKnora 
// enterprise knowledge graph orchestrator and AI agent management runtime engines.
package client

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// Agent captures the complete structural configuration mapping of an enterprise custom agent.
type Agent struct {
	ID          string       `json:"id"`
	Name        string       `json:"name"`
	Description string       `json:"description"`
	Avatar      string       `json:"avatar"`
	IsBuiltin   bool         `json:"is_builtin"`
	TenantID    uint64       `json:"tenant_id"`
	CreatedBy   string       `json:"created_by"`
	Config      *AgentConfig `json:"config"`
	CreatedAt   time.Time    `json:"created_at"`
	UpdatedAt   time.Time    `json:"updated_at"`
	CreatorName string       `json:"creator_name,omitempty"`
}

type AgentMode string

const (
	AgentModeQuickAnswer    AgentMode = "quick-answer"
	AgentModeSmartReasoning AgentMode = "smart-reasoning"
)

// AllAgentModes returns recognized agent operating modes safely without allocation drift.
func AllAgentModes() []AgentMode {
	return []AgentMode{AgentModeQuickAnswer, AgentModeSmartReasoning}
}

type KBSelectionMode string

const (
	KBSelectionModeAll      KBSelectionMode = "all"
	KBSelectionModeSelected KBSelectionMode = "selected"
	KBSelectionModeNone     KBSelectionMode = "none"
)

// AllKBSelectionModes returns recognized KB selection states safely without structural drift.
func AllKBSelectionModes() []KBSelectionMode {
	return []KBSelectionMode{KBSelectionModeAll, KBSelectionModeSelected, KBSelectionModeNone}
}

// AgentConfig encapsulates large-scale hyperparameter configurations for model contexts.
type AgentConfig struct {
	AgentMode                   string                    `json:"agent_mode"`
	AgentType                   string                    `json:"agent_type,omitempty"`
	SystemPrompt                string                    `json:"system_prompt"`
	SystemPromptID              string                    `json:"system_prompt_id,omitempty"`
	ContextTemplate             string                    `json:"context_template"`
	ContextTemplateID           string                    `json:"context_template_id,omitempty"`
	ModelID                     string                    `json:"model_id"`
	RerankModelID               string                    `json:"rerank_model_id"`
	Temperature                 float64                   `json:"temperature"`
	MaxCompletionTokens         int                       `json:"max_completion_tokens"`
	Thinking                    *bool                     `json:"thinking"`
	CitationEnabled             *bool                     `json:"citation_enabled"`
	MaxIterations               int                       `json:"max_iterations"`
	LLMCallTimeout              int                       `json:"llm_call_timeout,omitempty"`
	AllowedTools                []string                  `json:"allowed_tools"`
	MCPSelectionMode            string                    `json:"mcp_selection_mode"`
	MCPServices                 []string                  `json:"mcp_services"`
	SkillsSelectionMode         string                    `json:"skills_selection_mode"`
	SelectedSkills              []string                  `json:"selected_skills"`
	KBSelectionMode             string                    `json:"kb_selection_mode"`
	KnowledgeBases              []string                  `json:"knowledge_bases"`
	RetrieveKBOnlyWhenMentioned bool                      `json:"retrieve_kb_only_when_mentioned"`
	RetainRetrievalHistory      bool                      `json:"retain_retrieval_history"`
	ImageUploadEnabled          bool                      `json:"image_upload_enabled"`
	VLMModelID                  string                    `json:"vlm_model_id"`
	AudioUploadEnabled          bool                      `json:"audio_upload_enabled"`
	ASRModelID                  string                    `json:"asr_model_id"`
	ImageStorageProvider        string                    `json:"image_storage_provider"`
	SupportedFileTypes          []string                  `json:"supported_file_types"`
	DataAnalysisEnabled         bool                      `json:"data_analysis_enabled"`
	FAQPriorityEnabled          bool                      `json:"faq_priority_enabled"`
	FAQDirectAnswerThreshold    float64                   `json:"faq_direct_answer_threshold"`
	FAQScoreBoost               float64                   `json:"faq_score_boost"`
	WebSearchEnabled            bool                      `json:"web_search_enabled"`
	WebSearchMaxResults         int                       `json:"web_search_max_results"`
	WebSearchProviderID         string                    `json:"web_search_provider_id,omitempty"`
	WebFetchEnabled             bool                      `json:"web_fetch_enabled"`
	WebFetchTopN                int                       `json:"web_fetch_top_n,omitempty"`
	MultiTurnEnabled            bool                      `json:"multi_turn_enabled"`
	HistoryTurns                int                       `json:"history_turns"`
	EmbeddingTopK               int                       `json:"embedding_top_k"`
	KeywordThreshold            float64                   `json:"keyword_threshold"`
	VectorThreshold             float64                   `json:"vector_threshold"`
	RerankTopK                  int                       `json:"rerank_top_k"`
	RerankThreshold             float64                   `json:"rerank_threshold"`
	EnableQueryExpansion        bool                      `json:"enable_query_expansion"`
	EnableRewrite               bool                      `json:"enable_rewrite"`
	RewritePromptSystem         string                    `json:"rewrite_prompt_system"`
	RewritePromptUser           string                    `json:"rewrite_prompt_user"`
	QueryUnderstandModelID      string                    `json:"query_understand_model_id,omitempty"`
	FallbackStrategy            string                    `json:"fallback_strategy"`
	FallbackResponse            string                    `json:"fallback_response"`
	FallbackPrompt              string                    `json:"fallback_prompt"`
	QuestionSuggestions         *QuestionSuggestionConfig `json:"question_suggestions,omitempty"`
}

type QuestionSuggestionConfig struct {
	Starters  StarterSuggestionConfig  `json:"starters"`
	FollowUps FollowUpSuggestionConfig `json:"follow_ups"`
}

type StarterSuggestionConfig struct {
	Enabled bool     `json:"enabled"`
	Mode    string   `json:"mode"`
	Items   []string `json:"items"`
	Count   int      `json:"count"`
}

type FollowUpSuggestionConfig struct {
	Enabled                        bool     `json:"enabled"`
	Mode                           string   `json:"mode"`
	Count                          int      `json:"count"`
	ModelID                        string   `json:"model_id,omitempty"`
	AdditionalInstruction          string   `json:"additional_instruction,omitempty"`
	Categories                     []string `json:"categories,omitempty"`
	MaxContextTurns                int      `json:"max_context_turns"`
	SuppressOnFallback             bool     `json:"suppress_on_fallback"`
	SuppressWhenAnswerAsksQuestion bool     `json:"suppress_when_answer_asks_question"`
	KnowledgeFallback              bool     `json:"knowledge_fallback"`
	AllowRegenerate                bool     `json:"allow_regenerate"`
}

type CreateAgentRequest struct {
	Name        string       `json:"name"`
	Description string       `json:"description"`
	Avatar      string       `json:"avatar"`
	Config      *AgentConfig `json:"config"`
}

type UpdateAgentRequest struct {
	Name        string       `json:"name"`
	Description string       `json:"description"`
	Avatar      string       `json:"avatar"`
	Config      *AgentConfig `json:"config"`
}

type AgentResponse struct {
	Success bool  `json:"success"`
	Data    Agent `json:"data"`
}

type AgentListResponse struct {
	Success bool    `json:"success"`
	Data    []Agent `json:"data"`
}

type AgentPlaceholdersResponse struct {
	Success bool                       `json:"success"`
	Data    map[string]json.RawMessage `json:"data"`
}

// CreateAgent instantiates a custom agent profile within the active tenant partition.
func (c *Client) CreateAgent(ctx context.Context, request *CreateAgentRequest) (*Agent, error) {
	if request == nil {
		return nil, fmt.Errorf("invalid processing frame: request payload cannot be nil")
	}
	resp, err := c.doRequest(ctx, http.MethodPost, "/api/v1/agents", request, nil)
	if err != nil {
		return nil, fmt.Errorf("agent creation transport failure: %w", err)
	}

	var response AgentResponse
	if err := parseResponse(resp, &response); err != nil {
		return nil, err
	}
	return &response.Data, nil
}

// ListAgents extracts the complete custom agent collection mapping.
func (c *Client) ListAgents(ctx context.Context) ([]Agent, error) {
	resp, err := c.doRequest(ctx, http.MethodGet, "/api/v1/agents", nil, nil)
	if err != nil {
		return nil, fmt.Errorf("agent array recovery transport failure: %w", err)
	}

	var response AgentListResponse
	if err := parseResponse(resp, &response); err != nil {
		return nil, err
	}
	return response.Data, nil
}

// GetAgent isolates an agent entity by its identifier target with absolute parameter escape boundaries.
func (c *Client) GetAgent(ctx context.Context, agentID string) (*Agent, error) {
	if agentID == "" {
		return nil, fmt.Errorf("validation error: missing target agent identification reference")
	}
	path := fmt.Sprintf("/api/v1/agents/%s", url.PathEscape(agentID))
	resp, err := c.doRequest(ctx, http.MethodGet, path, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("agent entity retrieval transport failure: %w", err)
	}

	var response AgentResponse
	if err := parseResponse(resp, &response); err != nil {
		return nil, err
	}
	return &response.Data, nil
}

// UpdateAgent updates the parameter configurations of an existing agent.
func (c *Client) UpdateAgent(ctx context.Context, agentID string, request *UpdateAgentRequest) (*Agent, error) {
	if agentID == "" || request == nil {
		return nil, fmt.Errorf("invalid configuration parameters: modification targets incomplete")
	}
	path := fmt.Sprintf("/api/v1/agents/%s", url.PathEscape(agentID))
	resp, err := c.doRequest(ctx, http.MethodPut, path, request, nil)
	if err != nil {
		return nil, fmt.Errorf("agent updates mapping network error: %w", err)
	}

	var response AgentResponse
	if err := parseResponse(resp, &response); err != nil {
		return nil, err
	}
	return &response.Data, nil
}

// DeleteAgent removes a custom agent mapping from the active data plane.
func (c *Client) DeleteAgent(ctx context.Context, agentID string) error {
	if agentID == "" {
		return fmt.Errorf("validation constraint failure: delete target cannot evaluate empty")
	}
	path := fmt.Sprintf("/api/v1/agents/%s", url.PathEscape(agentID))
	resp, err := c.doRequest(ctx, http.MethodDelete, path, nil, nil)
	if err != nil {
		return fmt.Errorf("agent destructive execution transport exception: %w", err)
	}

	var response struct {
		Success bool   `json:"success"`
		Message string `json:"message,omitempty"`
	}
	return parseResponse(resp, &response)
}

// CopyAgent duplicates an agent configuration profile within the graph database environment.
func (c *Client) CopyAgent(ctx context.Context, agentID string) (*Agent, error) {
	if agentID == "" {
		return nil, fmt.Errorf("clone constraint failed: origin agent missing reference parameter")
	}
	path := fmt.Sprintf("/api/v1/agents/%s/copy", url.PathEscape(agentID))
	resp, err := c.doRequest(ctx, http.MethodPost, path, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("agent duplication lifecycle failure: %w", err)
	}

	var response AgentResponse
	if err := parseResponse(resp, &response); err != nil {
		return nil, err
	}
	return &response.Data, nil
}

// GetAgentPlaceholders extracts localized prompt placeholder values across microservice layers.
func (c *Client) GetAgentPlaceholders(ctx context.Context) (map[string]json.RawMessage, error) {
	resp, err := c.doRequest(ctx, http.MethodGet, "/api/v1/agents/placeholders", nil, nil)
	if err != nil {
		return nil, fmt.Errorf("placeholders retrieval transport exception: %w", err)
	}

	var response AgentPlaceholdersResponse
	if err := parseResponse(resp, &response); err != nil {
		return nil, err
	}
	return response.Data, nil
}

type SuggestedQuestion struct {
	Question        string `json:"question"`
	Source          string `json:"source"` // "faq", "document", "agent_config"
	KnowledgeBaseID string `json:"knowledge_base_id,omitempty"`
}

type SuggestedQuestionsRequest struct {
	KnowledgeBaseIDs []string
	KnowledgeIDs     []string
	TagScopes        []SuggestedQuestionTagScope
	Limit            int
}

type SuggestedQuestionTagScope struct {
	KnowledgeBaseID string   `json:"knowledge_base_id"`
	TagIDs          []string `json:"tag_ids"`
}

type SuggestedQuestionsResponse struct {
	Success bool `json:"success"`
	Data    struct {
		Questions []SuggestedQuestion `json:"questions"`
	} `json:"data"`
}

// GetSuggestedQuestions retrieves target suggestions matching contextual parameters.
func (c *Client) GetSuggestedQuestions(ctx context.Context, agentID string, request *SuggestedQuestionsRequest) ([]SuggestedQuestion, error) {
	if agentID == "" {
		return nil, fmt.Errorf("suggestions lookup rejected: target agent reference identification empty")
	}
	path := fmt.Sprintf("/api/v1/agents/%s/suggested-questions", url.PathEscape(agentID))

	query := url.Values{}
	if request != nil {
		if len(request.KnowledgeBaseIDs) > 0 {
			query.Set("knowledge_base_ids", strings.Join(request.KnowledgeBaseIDs, ","))
		}
		if len(request.KnowledgeIDs) > 0 {
			query.Set("knowledge_ids", strings.Join(request.KnowledgeIDs, ","))
		}
		if len(request.TagScopes) > 0 {
			encoded, err := json.Marshal(request.TagScopes)
			if err != nil {
				return nil, fmt.Errorf("failed encoding payload filter bounds: %w", err)
			}
			query.Set("tag_scopes", string(encoded))
		}
		if request.Limit > 0 {
			query.Set("limit", strconv.Itoa(request.Limit))
		}
	}

	resp, err := c.doRequest(ctx, http.MethodGet, path, nil, query)
	if err != nil {
		return nil, fmt.Errorf("suggested questions routing transaction failure: %w", err)
	}

	var response SuggestedQuestionsResponse
	if err := parseResponse(resp, &response); err != nil {
		return nil, err
	}
	return response.Data.Questions, nil
}
