<script lang="ts">
    import { getSemanticBoundaryState } from "../js/stores/semanticBoundaryStore";

    interface MyProps {
        messageId: string;
    }
    export let messageId: string;
    
    const semanticBoundaryState = getSemanticBoundaryState();
    
    // Calculate derived values using standard Svelte reactivity
    $: distance = semanticBoundaryState.getDistance(messageId);
    $: justification = semanticBoundaryState.getJustification(messageId);
    $: hasDistance = distance > 0; // Only show if there's an actual distance
    $: isSignificantBoundary = distance >= 0.4; // Significant boundaries
    
    // Process justification to highlight key points
    $: formattedJustification = formatJustification(justification);
    
    // Green-to-red color gradient (low to high distance)
    $: boundaryColor = generateGreenToRedColor(distance * 2); // Scale up for better differentiation
    
    // Higher distances get more visible separators
    $: lineHeight = Math.max(1, Math.min(4, Math.floor(distance * 8)));
    $: opacity = Math.min(1, distance * 2);
    
    // Format justification into key metrics and reasons
    function formatJustification(text: string): {key: string, detail: string} {
        if (!text) return { key: '', detail: '' };
        
        // Extract the most important part of the justification for the key
        let key = '';
        if (text.includes('Significant semantic boundary') || 
            text.includes('Major time boundary')) {
            key = 'Major boundary';
        } else if (text.includes('Significant time boundary')) {
            key = 'Time boundary';
        } else if (text.includes('Strong depth')) {
            key = 'Topic shift';
        } else if (text.includes('High semantic')) {
            key = 'Content change';
        } else if (text.includes('Time gap') && text.includes('h') || text.includes('d')) {
            key = 'Time gap';
        } else {
            key = 'Minor change';
        }
        
        // Clean up the detail text
        // Replace verbose phrases with shorter versions
        let detail = text
            // .replace('Significant semantic boundary', '🔀 Topic boundary')
            // .replace('Significant time boundary', '⏱️ Time boundary')
            // .replace('Major time boundary', '⏱️ Major time gap')
            // .replace('High semantic difference', 'High difference')
            // .replace('Moderate semantic difference', 'Moderate difference')
            // .replace('Strong depth score', 'Strong depth')
            // .replace('Moderate depth score', 'Medium depth')
            // .replace('Minor depth score', 'Small depth');
            
        return { key, detail };
    }
    
    /**
     * Generate a color on a green-to-red scale
     * Green = Low distance (little change)
     * Red = High distance (significant topic change)
     * @param value Value between 0-1
     * @returns CSS color string
     */
    function generateGreenToRedColor(value: number): string {
        // Ensure the value is between 0 and 1
        const t = Math.max(0, Math.min(1, value));
        
        // Green to red gradient
        const r = Math.floor(t * 255);         // Red increases with distance
        const g = Math.floor(255 * (1 - t));   // Green decreases with distance
        const b = 0;                           // No blue component
        
        return `rgb(${r}, ${g}, ${b})`;
    }
</script>

{#if hasDistance}
<div class="semantic-boundary" style="opacity: {opacity}">
    <div class="semantic-boundary-line" style="background-color: {boundaryColor}; height: {lineHeight}px"></div>
    <div class="semantic-boundary-content">
        {#if isSignificantBoundary}
            <!-- For significant boundaries, show full justification -->
            <div class="semantic-boundary-info significant">
                <div class="info-row">
                    <span class="score">{distance.toFixed(2)}</span>
                    <span class="boundary-type">{formattedJustification.key}</span>
                </div>
                {#if formattedJustification.detail}
                    <div class="justification">{formattedJustification.detail}</div>
                {/if}
            </div>
        {:else if justification}
            <!-- For minor boundaries, show compact info -->
            <div class="semantic-boundary-info minor">
                <div class="info-row">
                    <span class="score">{distance.toFixed(2)}</span>
                    {#if formattedJustification.key}
                        <span class="boundary-type minor-type">{formattedJustification.key}</span>
                    {/if}
                </div>
            </div>
        {:else}
            <!-- Fallback if no justification -->
            <div class="semantic-boundary-distance">{distance.toFixed(2)}</div>
        {/if}
    </div>
    <div class="semantic-boundary-line" style="background-color: {boundaryColor}; height: {lineHeight}px"></div>
</div>
{/if}

<style>
    .semantic-boundary {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 4px 15px;
        gap: 5px;
    }
    
    .semantic-boundary-line {
        width: 100%;
        height: 1px;
    }
    
    .semantic-boundary-content {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    .semantic-boundary-distance {
        color: #d8d8d8;
        font-size: 10px;
        white-space: nowrap;
        padding: 0 5px;
    }
    
    /* Styling for justification info */
    .semantic-boundary-info {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 2px 4px;
        border-radius: 4px;
    }
    
    .semantic-boundary-info.significant {
        background-color: rgba(30, 30, 30, 0.85);
        padding: 3px 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
        margin: 2px 0;
    }
    
    .info-row {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .score {
        color: #ffffff;
        font-size: 10px;
        font-weight: 500;
    }
    
    .boundary-type {
        color: #ffcc44;
        font-size: 10px;
        font-weight: 600;
    }
    
    .minor-type {
        color: #b8b8b8;
        font-weight: normal;
    }
    
    .justification {
        color: #e0e0e0;
        font-size: 9px;
        max-width: 400px;
        text-align: center;
        white-space: normal;
        line-height: 1.3;
        margin-top: 2px;
    }
</style>
