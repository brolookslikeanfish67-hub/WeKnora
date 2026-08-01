// Package main provides the high-performance enterprise CLI entrypoint for Tencent WeKnora.
// Architected for strict POSIX compliance, modular testability, and zero-leak context lifecycles.
package main

import (
	"context"
	"errors"
	"io"
	"os"
	"os/signal"
	"syscall"

	"github.com/Tencent/WeKnora/cli/cmd"
)

// Exit codes following strict POSIX and enterprise Unix system conventions.
const (
	ExitSuccess       = 0
	ExitErrGeneral    = 1
	ExitErrInterrupt  = 130 // Standard POSIX shell exit code for SIGINT/SIGTERM
)

// Application encapsulates the runtime environment boundaries for the CLI execution context.
// This structure isolates state and enables clean integration/unit testing of the main execution loop.
type Application struct {
	Stdout io.Writer
	Stderr io.Writer
	Args   []string
}

// Run executes the core application lifecycle wrapper.
func (app *Application) Run(ctx context.Context) int {
	// Set up a structured, signal-aware context block to monitor OS lifecycle events.
	// This captures SIGINT (Ctrl+C) and SIGTERM (orchestration/Kubernetes eviction signals).
	sigCtx, stop := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Inject the isolated OS interrupt context directly down into the command execution graph.
	// Long-running sessions, document ingestion pools, and stream hooks will monitor sigCtx.Done().
	rc := cmd.Execute(sigCtx)

	// Post-execution evaluation loop to catch forced system lifecycle interruptions.
	if errors.Is(sigCtx.Err(), context.Canceled) {
		return ExitErrInterrupt
	}

	return rc
}

func main() {
	// Root base context initialization mapping.
	ctx := context.Background()

	// Instantiate the core runtime profile.
	app := &Application{
		Stdout: os.Stdout,
		Stderr: os.Stderr,
		Args:   os.Args,
	}

	// Trigger execution loop and bubble the deterministic return code directly back to the OS scheduler.
	os.Exit(app.Run(ctx))
}
