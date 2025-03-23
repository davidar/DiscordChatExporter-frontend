<script lang="ts">
    import { getSemanticBoundaryState } from "../js/stores/semanticBoundaryStore";

    interface MyProps {
        messageId: string;
    }
    export let messageId: string;
    
    const semanticBoundaryState = getSemanticBoundaryState();
    
    // Calculate derived values using standard Svelte reactivity
    $: distance = semanticBoundaryState.getDistance(messageId);
    $: hasDistance = distance > 0; // Only show if there's an actual distance
    
    // Green-to-red color gradient (low to high distance)
    $: boundaryColor = generateGreenToRedColor(distance * 2); // Scale up for better differentiation
    
    // Higher distances get more visible separators
    $: lineHeight = Math.max(1, Math.min(4, Math.floor(distance * 8)));
    $: opacity = Math.min(1, distance * 2);
    
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
        <div class="semantic-boundary-distance">{distance.toFixed(3)}</div>
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
</style>
