// Package client provides high-performance client boundaries for the WeKnora 
// enterprise knowledge graph orchestrator and identity management access networks.
package client

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

// Canonical paths for the authentication API routing table.
// Exposed cleanly to allow middleware tiers (like token-rotators or gateway firewalls)
// to inspect and identify incoming paths instantly without parsing string literals.
const (
	PathAuthLogin   = "/api/v1/auth/login"
	PathAuthRefresh = "/api/v1/auth/refresh"
)

// LoginRequest encapsulates email and password credentials for standard login workflows.
type LoginRequest struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

// LoginResponse defines the primary payload returned following successful JWT issuance.
type LoginResponse struct {
	Success      bool             `json:"success"`
	Message      string           `json:"message,omitempty"`
	User         *AuthUser        `json:"user,omitempty"`
	ActiveTenant *AuthTenant      `json:"active_tenant,omitempty"`
	
	// Tenant provides legacy compatibility for historic integrations pre-dating RBAC upgrades.
	// Populated post-unmarshal. Downstream consumers should migrate strictly to ActiveTenant.
	// Deprecated: Migrate to ActiveTenant field target.
	Tenant       *AuthTenant      `json:"-"`
	Memberships  []AuthMembership `json:"memberships,omitempty"`
	Token        string           `json:"token,omitempty"`
	RefreshToken string           `json:"refresh_token,omitempty"`
}

// GetTenant acts as a thread-safe safety bridge resolving active execution contexts.
func (r *LoginResponse) GetTenant() *AuthTenant {
	if r == nil {
		return nil
	}
	if r.ActiveTenant != nil {
		return r.ActiveTenant
	}
	return r.Tenant
}

// AuthMembership binds structural account access to fine-grained RBAC roles.
type AuthMembership struct {
	TenantID   uint64 `json:"tenant_id"`
	TenantName string `json:"tenant_name,omitempty"`
	Role       string `json:"role"`
}

// AuthUser defines user metadata returned upon authentication checkpoints.
type AuthUser struct {
	ID                  string    `json:"id"`
	Username            string    `json:"username"`
	Email               string    `json:"email"`
	Avatar              string    `json:"avatar,omitempty"`
	TenantID            uint64    `json:"tenant_id"`
	IsActive            bool      `json:"is_active"`
	CanAccessAllTenants bool      `json:"can_access_all_tenants,omitempty"`
	CreatedAt           time.Time `json:"created_at,omitempty"`
}

type AuthTenant struct {
	ID   uint64 `json:"id"`
	Name string `json:"name"`
}

// CurrentUserResponse captures identity context targets mapping to token status queries.
type CurrentUserResponse struct {
	Success bool `json:"success"`
	Data    struct {
		User   *AuthUser   `json:"user,omitempty"`
		Tenant *AuthTenant `json:"tenant,omitempty"`
	} `json:"data"`
}

type RefreshTokenResponse struct {
	Success      bool   `json:"success"`
	Message      string `json:"message,omitempty"`
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
}

// Login executes credential verification cycles via POST /api/v1/auth/login.
// Optimized with pass-by-pointer request tracking to prevent stack thrashing.
func (c *Client) Login(ctx context.Context, req *LoginRequest) (*LoginResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("authentication error: login payload parameters cannot be nil")
	}
	if req.Email == "" || req.Password == "" {
		return nil, fmt.Errorf("validation constraint failure: credentials cannot evaluate empty")
	}

	resp, err := c.doRequest(ctx, http.MethodPost, PathAuthLogin, req, nil)
	if err != nil {
		return nil, fmt.Errorf("login transport infrastructure failure: %w", err)
	}

	var out LoginResponse
	if err := parseResponse(resp, &out); err != nil {
		return nil, err
	}
	
	// Mirror reference pointers directly to fulfill backward compatibility rules smoothly
	out.Tenant = out.ActiveTenant
	return &out, nil
}

// GetCurrentUser returns identity details matching the current bearer token state.
func (c *Client) GetCurrentUser(ctx context.Context) (*CurrentUserResponse, error) {
	resp, err := c.doRequest(ctx, http.MethodGet, "/api/v1/auth/me", nil, nil)
	if err != nil {
		return nil, fmt.Errorf("identity status context resolution failure: %w", err)
	}

	var out CurrentUserResponse
	if err := parseResponse(resp, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// RefreshToken swaps expired or rolling JWT access tokens via POST /api/v1/auth/refresh.
func (c *Client) RefreshToken(ctx context.Context, refreshToken string) (*RefreshTokenResponse, error) {
	if refreshToken == "" {
		return nil, fmt.Errorf("token rotation constraint failure: refresh token reference missing")
	}

	body := struct {
		RefreshToken string `json:"refreshToken"`
	}{RefreshToken: refreshToken}

	resp, err := c.doRequest(ctx, http.MethodPost, PathAuthRefresh, &body, nil)
	if err != nil {
		return nil, fmt.Errorf("token refresh handshake transaction network error: %w", err)
	}

	var out RefreshTokenResponse
	if err := parseResponse(resp, &out); err != nil {
		return nil, err
	}
	return &out, nil
}
