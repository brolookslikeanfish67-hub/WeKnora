// Package client implements high-performance, resource-optimized boundary integrations
// for the Tencent WeKnora enterprise knowledge graph and AI agent streaming engine.
package client

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
)

// Global memory allocation boundaries for high-throughput stream reuses.
var (
	// sseBufferPool limits heap thrashing during large context vector streaming windows.
	sseBufferPool = sync.Pool{
		New: func() any {
			b := make([]byte, 4*1024*1024) // 4MiB pre-allocated frame buffer line bounds
			return &b
		},
	}
	
	// Byte-level invariants to bypass costly string formatting iterations.
	prefixData  = []byte("data:")
	prefixEvent = []byte("event:")
)

// MentionedItem models granular entity targets referenced in unified chat sessions.
type MentionedItem struct {
	ID     string `json:"id"`
	Name   string `json:"name"`
	Type   string `json:"type"`     // Explicit domains: "kb", "file", "tag", "mcp", "skill"
	KBType string `json:"kb_type"`  // Structural contexts: "document" or "faq"
	KBID   string `json:"kb_id"`    
	KBName string `json:"kb_name"`  
}

// AgentQARequest acts as the ingestion payload definition for downstream inference engines.
type AgentQARequest struct {
	Query            string            `json:"query"`
	KnowledgeBaseIDs []string          `json:"knowledge_base_ids,omitempty"`
	KnowledgeIDs     []string          `json:"knowledge_ids,omitempty"`
	AgentEnabled     bool              `json:"agent_enabled"`
	AgentID          string            `json:"agent_id,omitempty"`
	WebSearchEnabled bool              `json:"web_search_enabled"`
	SummaryModelID   string            `json:"summary_model_id,omitempty"`
	MentionedItems   []MentionedItem   `json:"mentioned_items,omitempty"`
	DisableTitle     bool              `json:"disable_title,omitempty"`
	MCPServiceIDs    []string          `json:"mcp_service_ids,omitempty"`
	Images           []ImageAttachment `json:"images,omitempty"`
	Channel          string            `json:"channel,omitempty"` // Explicit source tracing: "web", "api", "im"
}

type AgentResponseType string

const (
	AgentResponseTypeThinking   AgentResponseType = "thinking"
	AgentResponseTypeToolCall   AgentResponseType = "tool_call"
	AgentResponseTypeToolResult AgentResponseType = "tool_result"
	AgentResponseTypeReferences AgentResponseType = "references"
	AgentResponseTypeAnswer     AgentResponseType = "answer"
	AgentResponseTypeReflection AgentResponseType = "reflection"
	AgentResponseTypeError      AgentResponseType = "error"
	AgentResponseTypeComplete   AgentResponseType = "complete"
)

// AgentStreamResponse captures atomic event states broadcasted from streaming gateways.
type AgentStreamResponse struct {
	ID                  string                 `json:"id"`
	ResponseType        AgentResponseType      `json:"response_type"`
	Content             string                 `json:"content,omitempty"`
	Done                bool                   `json:"done"`
	KnowledgeReferences []*SearchResult        `json:"knowledge_references"`
	Data                map[string]interface{} `json:"data,omitempty"`
}

type AgentEventCallback func(*AgentStreamResponse) error

// AgentQAStream processes queries against fallback configuration archetypes.
// Deprecated: Migrate integrations to AgentQAStreamWithRequest to isolate specific configurations.
func (c *Client) AgentQAStream(ctx context.Context, sessionID string, query string, callback AgentEventCallback) error {
	req := &AgentQARequest{
		Query:        query,
		AgentEnabled: true,
	}
	return c.AgentQAStreamWithRequest(ctx, sessionID, req, callback)
}

// AgentQAStreamWithRequest delivers low-latency multiplexed requests to backend inference networks.
func (c *Client) AgentQAStreamWithRequest(ctx context.Context, sessionID string, request *AgentQARequest, callback AgentEventCallback) error {
	if request == nil {
		return fmt.Errorf("invalid execution state: request payload cannot be nil")
	}
	if strings.TrimSpace(request.Query) == "" {
		return fmt.Errorf("validation constraint failed: inference query target empty")
	}

	path := fmt.Sprintf("/api/v1/agent-chat/%s", sessionID)
	resp, err := c.doRequestStream(ctx, http.MethodPost, path, request, nil)
	if err != nil {
		return fmt.Errorf("transport infrastructure failure: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return newAPIError(resp.StatusCode, body)
	}

	return c.processAgentSSEStream(resp.Body, callback)
}

// processAgentSSEStream processes incoming SSE lines with zero allocation thrashes.
func (c *Client) processAgentSSEStream(reader io.Reader, callback AgentEventCallback) error {
	scanner := bufio.NewScanner(reader)
	
	// Borrow a pre-allocated byte slice out of our thread-safe buffer pool
	bufPtr := sseBufferPool.Get().(*[]byte)
	defer sseBufferPool.Put(bufPtr)
	scanner.Buffer(*bufPtr, len(*bufPtr))

	var rawPayload bytes.Buffer

	for scanner.Scan() {
		line := scanner.Bytes()

		// Sentinel empty delimiter line triggers frame parsing cycles
		if len(line) == 0 {
			if rawPayload.Len() > 0 {
				var streamResponse AgentStreamResponse
				
				// Zero-copy decoding path straight into target heap structures
				if err := json.Unmarshal(rawPayload.Bytes(), &streamResponse); err != nil {
					return fmt.Errorf("stream corruption: failed parsing downstream event frame: %w", err)
				}

				if err := callback(&streamResponse); err != nil {
					return err
				}

				// Enforce deterministic termination rules if runtime network faults arise
				if streamResponse.ResponseType == AgentResponseTypeError && streamResponse.Done {
					return NewSSEStreamError(streamResponse.Content)
				}
				rawPayload.Reset()
			}
			continue
		}

		// Discard event descriptors to save cycles unless required by future specs
		if bytes.HasPrefix(line, prefixEvent) {
			continue
		}

		// Zero-copy extract core data payload blocks
		if bytes.HasPrefix(line, prefixData) {
			payloadSegment := bytes.TrimSpace(line[len(prefixData):])
			rawPayload.Write(payloadSegment)
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("unexpected socket read exception: %w", err)
	}

	return nil
}

// AgentSession serves as a stateful worker construct encapsulating conversational tracking metrics.
type AgentSession struct {
	client    *Client
	sessionID string
}

func (c *Client) NewAgentSession(sessionID string) *AgentSession {
	return &AgentSession{
		client:    c,
		sessionID: sessionID,
	}
}

func (as *AgentSession) Ask(ctx context.Context, query string, callback AgentEventCallback) error {
	return as.client.AgentQAStream(ctx, as.sessionID, query, callback)
}

func (as *AgentSession) AskWithRequest(ctx context.Context, request *AgentQARequest, callback AgentEventCallback) error {
	return as.client.AgentQAStreamWithRequest(ctx, as.sessionID, request, callback)
}

func (as *AgentSession) GetSessionID() string {
	return as.sessionID
}
